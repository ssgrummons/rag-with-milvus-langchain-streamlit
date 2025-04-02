import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from pymilvus.exceptions import MilvusException
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import import_from_string
from keybert import KeyBERT
import markdown
import frontmatter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dataclasses import dataclass
from enum import Enum
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class MilvusError(Exception):
    """Custom exception for Milvus-related errors."""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

@dataclass
class Config:
    """Configuration class with validation."""
    embedding_model: str
    keyword_model: str
    chunk_size: int
    chunk_overlap: int
    markdown_folder: str
    milvus_host: str
    milvus_port: str
    collection_name: str
    collection_schema: Dict[str, Any]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary with validation."""
        try:
            # Validate required fields
            required_fields = [
                'embedding_model', 'keyword_model', 'chunk_size', 
                'chunk_overlap', 'markdown_folder', 'milvus', 'collection'
            ]
            for field in required_fields:
                if field not in config_dict:
                    raise ConfigError(f"Missing required field: {field}")

            # Validate numeric fields
            if not isinstance(config_dict['chunk_size'], int) or config_dict['chunk_size'] <= 0:
                raise ConfigError("chunk_size must be a positive integer")
            if not isinstance(config_dict['chunk_overlap'], int) or config_dict['chunk_overlap'] < 0:
                raise ConfigError("chunk_overlap must be a non-negative integer")

            # Validate markdown folder exists
            if not os.path.exists(config_dict['markdown_folder']):
                raise ConfigError(f"markdown_folder does not exist: {config_dict['markdown_folder']}")

            # Validate collection schema
            if 'schema' not in config_dict['collection']:
                raise ConfigError("collection schema is missing")

            return cls(
                embedding_model=config_dict['embedding_model'],
                keyword_model=config_dict['keyword_model'],
                chunk_size=config_dict['chunk_size'],
                chunk_overlap=config_dict['chunk_overlap'],
                markdown_folder=config_dict['markdown_folder'],
                milvus_host=config_dict['milvus']['host'],
                milvus_port=str(config_dict['milvus']['port']),
                collection_name=config_dict['collection']['name'],
                collection_schema=config_dict['collection']['schema']
            )
        except Exception as e:
            raise ConfigError(f"Invalid configuration: {str(e)}")

class MilvusConnector:
    def __init__(self, config: Config):
        self.config = config
        try:
            # Initialize models
            logger.info(f"Loading embedding model: {config.embedding_model}")
            self.embedding_model = SentenceTransformer(config.embedding_model)
            
            logger.info(f"Loading keyword model: {config.keyword_model}")
            self.keyword_model = KeyBERT(config.keyword_model)
            
            # Connect to Milvus
            logger.info(f"Connecting to Milvus at {config.milvus_host}:{config.milvus_port}")
            connections.connect("default", host=config.milvus_host, port=config.milvus_port)
        except Exception as e:
            raise ModelError(f"Failed to initialize models or connect to Milvus: {str(e)}")

    def _create_field_schema(self, field_config: Dict[str, Any]) -> FieldSchema:
        """Create a Milvus FieldSchema from config."""
        try:
            data_type = getattr(DataType, field_config["data_type"])            
            field_params = {
                "name": field_config["name"],
                "dtype": data_type,
                "description": field_config["description"],
                "is_primary": field_config.get("is_primary", False),
                "auto_id": field_config.get("auto_id", False),
                "required": field_config.get("required", False)
            }
            if data_type == DataType.FLOAT_VECTOR:
                field_params["dim"] = field_config["dim"]            
            if data_type == DataType.VARCHAR:
                field_params["max_length"] = field_config.get("max_length", 65535)            
            if data_type == DataType.ARRAY:
                element_type = getattr(DataType, field_config["element_type"])
                field_params["element_type"] = element_type
                field_params["max_capacity"] = field_config.get("max_capacity", 100)
                if element_type == DataType.VARCHAR:
                    field_params["max_length"] = field_config.get("max_length", 65535)              
            return FieldSchema(**field_params)        
        except Exception as e:
            raise ConfigError(f"Invalid field schema: {str(e)}")


    def _create_collection_schema(self) -> CollectionSchema:
        """Create a Milvus CollectionSchema from config."""
        try:
            fields = [
                self._create_field_schema(field)
                for field in self.config.collection_schema["fields"]
            ]
            
            return CollectionSchema(
                fields=fields,
                description=f"Collection for {self.config.collection_name}"
            )
        except Exception as e:
            raise ConfigError(f"Invalid collection schema: {str(e)}")

    def ensure_collection_exists(self) -> None:
        """Ensure collection exists with correct schema."""
        try:
            if utility.has_collection(self.config.collection_name):
                logger.info(f"Collection {self.config.collection_name} exists, checking schema...")
                collection = Collection(name=self.config.collection_name)
                collection.load()
                
                existing_schema = collection.schema
                new_schema = self._create_collection_schema()
                
                if str(existing_schema) != str(new_schema):
                    logger.warning(f"Collection {self.config.collection_name} has different schema. Recreating...")
                    utility.drop_collection(self.config.collection_name)
                    collection = Collection(name=self.config.collection_name, schema=new_schema)
                    collection.create_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2")
            else:
                logger.info(f"Creating new collection: {self.config.collection_name}")
                schema = self._create_collection_schema()
                collection = Collection(name=self.config.collection_name, schema=schema)
                collection.create_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2")
        except MilvusException as e:
            raise MilvusError(f"Failed to manage collection: {str(e)}")

    def insert_data(self, chunks: List[str], metadata_list: List[Dict]) -> None:
        """Insert data into Milvus collection."""
        try:
            collection = Collection(name=self.config.collection_name)
            collection.load()
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Generate keywords
            logger.info("Generating keywords...")
            keywords_list = []
            for chunk in chunks:
                keywords = self.keyword_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 3))
                keywords_list.append([k[0] for k in keywords])
            
            # Prepare entities
            entities = []
            for i in range(len(chunks)):
                entity = {
                    "embedding": embeddings[i],
                    "content": chunks[i],
                    "metadata": metadata_list[i],
                    "keywords": keywords_list[i],
                    "created_at": datetime.now().isoformat()
                }
                entities.append(entity)
            
            # Insert data
            logger.info(f"Inserting {len(entities)} entities...")
            collection.insert(entities)
            collection.flush()
            logger.info(f"Successfully inserted {len(entities)} entries into {self.config.collection_name}")
        except MilvusException as e:
            raise MilvusError(f"Failed to insert data into Milvus: {str(e)}")
        except Exception as e:
            raise ModelError(f"Failed to process data: {str(e)}")

class MarkdownProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def parse_markdown(file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
                content = post.content            
            parsed_content = markdown.markdown(content)
            headers = {}
            current_level = 1            
            while True:
                level_str = f"header.{current_level}"
                if level_str in parsed_content and isinstance(parsed_content[level_str], str):
                    headers[level_str] = parsed_content[level_str]
                    current_level += 1
                else:
                    break            
            metadata = headers 
            return metadata, parsed_content
        except Exception as e:
            raise ConfigError(f"Failed to parse markdown file {file_path}: {str(e)}")

    def chunk_text(self, text: str):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
        except Exception as e:
            raise ModelError(f"Failed to chunk text: {str(e)}")

    def process_markdown(self, file_path: str) -> List[Dict]:
        """Process a markdown file and return chunks with metadata."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into lines for processing
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_headers = {}  # Track current header hierarchy
        current_level = 0
        
        for line in lines:
            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_headers[str(level)] = title
                # Remove lower level headers when we encounter a higher level
                current_headers = {k: v for k, v in current_headers.items() if int(k) <= level}
                continue
            
            # Add line to current chunk
            current_chunk.append(line)
            
            # Check if we should create a new chunk
            if len('\n'.join(current_chunk)) >= self.chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'source': file_path,
                            'headers': current_headers.copy()  # Include current header hierarchy
                        }
                    })
                current_chunk = []
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'source': file_path,
                        'headers': current_headers.copy()  # Include current header hierarchy
                    }
                })
        
        return chunks

def load_config(config_path="config.yaml") -> Config:
    """Load and validate configuration."""
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config.from_dict(config_dict)
    except Exception as e:
        raise ConfigError(f"Failed to load config file: {str(e)}")

def process_documents(config: Config, milvus_client: MilvusConnector) -> None:
    """Process markdown documents and insert them into Milvus."""
    try:
        # Ensure collection exists with correct schema
        milvus_client.ensure_collection_exists()
        
        # Process markdown files
        markdown_files = list(Path(config.markdown_folder).rglob("*.md"))
        if not markdown_files:
            logger.warning(f"No markdown files found in {config.markdown_folder}")
            return

        # Initialize MarkdownProcessor with config values
        markdown_processor = MarkdownProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        for file in markdown_files:
            try:
                logger.info(f"Processing {file}...")
                chunks = markdown_processor.process_markdown(str(file))
                milvus_client.insert_data([chunk['content'] for chunk in chunks], [chunk['metadata'] for chunk in chunks])
            except Exception as e:
                logger.error(f"Failed to process file {file}: {str(e)}")
                continue
    except Exception as e:
        raise MilvusError(f"Failed to process documents: {str(e)}")

def main():
    try:
        # Load and validate config
        config = load_config()
        
        # Initialize Milvus connector
        milvus_client = MilvusConnector(config)
        
        # Process documents
        process_documents(config, milvus_client)
        
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        exit(1)
    except MilvusError as e:
        logger.error(f"Milvus error: {str(e)}")
        exit(1)
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    main()