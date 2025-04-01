# DataNinja Query Guide

This guide covers the various query methods available in DataNinja, from basic SQL queries to advanced analytics operations.

## SQL Queries

### Basic Queries
```sql
-- Simple SELECT query
SELECT * FROM users WHERE age > 25;

-- Aggregation with GROUP BY
SELECT department, COUNT(*) as employee_count 
FROM employees 
GROUP BY department;

-- JOIN operations
SELECT o.order_id, c.customer_name 
FROM orders o 
JOIN customers c ON o.customer_id = c.id;
```

### Advanced Queries
```sql
-- Window functions
SELECT 
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;

-- Common Table Expressions (CTEs)
WITH top_customers AS (
    SELECT customer_id, SUM(order_amount) as total_spent
    FROM orders
    GROUP BY customer_id
    ORDER BY total_spent DESC
    LIMIT 10
)
SELECT c.*, tc.total_spent
FROM customers c
JOIN top_customers tc ON c.id = tc.customer_id;
```

## GraphQL Queries

### Basic Queries
```graphql
query {
  users {
    id
    name
    email
    department
  }
}
```

### Complex Queries
```graphql
query {
  orders {
    id
    customer {
      name
      email
    }
    items {
      product {
        name
        price
      }
      quantity
    }
    totalAmount
    status
  }
}
```

## Stream Processing Queries

### Real-time Analytics
```sql
-- Stream processing query
SELECT 
    window_start,
    COUNT(*) as event_count,
    AVG(value) as avg_value
FROM event_stream
GROUP BY TUMBLE(interval '5' minute);
```

### Pattern Matching
```sql
-- Complex event processing
SELECT 
    pattern_start,
    pattern_end,
    matched_events
FROM event_stream
MATCH_RECOGNIZE (
    PARTITION BY device_id
    ORDER BY timestamp
    MEASURES
        FIRST(timestamp) as pattern_start,
        LAST(timestamp) as pattern_end,
        COUNT(*) as matched_events
    PATTERN (A B+ C)
    DEFINE
        A AS temperature > 80,
        B AS temperature > 75,
        C AS temperature < 70
);
```

## Natural Language Queries

### Business Intelligence
```text
Show me the sales trend for product category "Electronics" 
over the last 6 months, broken down by region.
```

### Data Exploration
```text
Find all customers who made purchases above $1000 
in the last month and their total spending.
```

## Query Optimization

### Best Practices
1. Use appropriate indexes
2. Limit result sets
3. Optimize JOIN operations
4. Use materialized views for complex queries
5. Implement query caching

### Performance Tips
- Use EXPLAIN to analyze query plans
- Monitor query execution times
- Implement query timeouts
- Use batch operations for bulk data

## Query Security

### Access Control
```sql
-- Role-based access example
GRANT SELECT ON sales_data TO analyst_role;
GRANT INSERT ON sales_data TO data_entry_role;
```

### Data Masking
```sql
-- Sensitive data masking
CREATE VIEW masked_customers AS
SELECT 
    id,
    name,
    MASK(email) as masked_email,
    MASK(phone) as masked_phone
FROM customers;
```

## Error Handling

### Common Errors
1. Syntax errors
2. Permission denied
3. Resource limits exceeded
4. Connection timeouts
5. Invalid data types

### Error Recovery
- Implement retry logic
- Use transaction rollbacks
- Log error details
- Monitor error patterns

## Query Monitoring

### Performance Metrics
- Query execution time
- Resource utilization
- Cache hit rates
- Error rates

### Monitoring Tools
- Query profiler
- Performance dashboard
- Alert system
- Audit logs
