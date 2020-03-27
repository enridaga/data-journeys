


from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "openaq" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there's only one!)

for table in tables:  

    print(table.table_id)


# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to select all the items from the "city" column where the "country" column is 'US'

query = """

        SELECT city

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
# Create a "Client" object

client = bigquery.Client()
# Set up the query

query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame

us_cities = query_job.to_dataframe()
# What five cities have the most measurements?

us_cities.city.value_counts().head()
query = """

        SELECT city, country

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
# Query to get the score column from every row where the type column has value "job"

query = """

        SELECT score, title

        FROM `bigquery-public-data.hacker_news.full`

        WHERE type = "job" 

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
# Only run the query if it's less than 1 MB

ONE_MB = 1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)



# Set up the query (will only run if it's less than 1 MB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_query_job.to_dataframe()
# Only run the query if it's less than 1 GB

ONE_GB = 1000*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)



# Set up the query (will only run if it's less than 1 GB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

job_post_scores = safe_query_job.to_dataframe()



# Print average score for job posts

job_post_scores.score.mean()