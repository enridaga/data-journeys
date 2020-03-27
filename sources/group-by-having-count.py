


from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "comments" table

table_ref = dataset_ref.table("comments")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to select comments that received more than 10 replies

query_popular = """

                SELECT parent, COUNT(id)

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_popular, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



# Print the first five rows of the DataFrame

popular_comments.head()
# Improved version of earlier query, now with aliasing & improved readability

query_improved = """

                 SELECT parent, COUNT(1) AS NumPosts

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY parent

                 HAVING COUNT(1) > 10

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()
query_good = """

             SELECT parent, COUNT(id)

             FROM `bigquery-public-data.hacker_news.comments`

             GROUP BY parent

             """
query_bad = """

            SELECT author, parent, COUNT(id)

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            """