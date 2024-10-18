import ssl
import socket
import whois


def check_ssl_certificate(url):
    # Extract the hostname from the URL
    hostname = url.split("://")[-1]

    # Create a context for SSL connections
    context = ssl.create_default_context()

    # Set up a secure connection to the server
    with context.wrap_socket(socket.socket(), server_hostname=hostname) as conn:
        conn.settimeout(5)  # Set a timeout for the connection
        try:
            # Connect to the server on port 443 (HTTPS)
            conn.connect((hostname, 443))

            # Get the server certificate
            cert = conn.getpeercert()

            # If the certificate is empty, the site does not have an SSL certificate
            if cert:
                print(f"The website '{url}' has an SSL certificate.")
            else:
                print(f"The website '{url}' does not have an SSL certificate.")

        except socket.timeout:
            print(f"Connection to '{url}' timed out.")

        except Exception as e:
            print(f"An error occurred while checking SSL certificate: {e}")


def get_whois_info(domain):
    # Perform a WHOIS query for the given domain
    domain_info = whois.whois(domain)

    # Check if the query was successful
    if domain_info:
        # Display domain information
        print(f"Domain Name: {domain_info.domain_name}")
        print(f"Registrar: {domain_info.registrar}")
        print(f"Creation Date: {domain_info.creation_date}")
        print(f"Expiration Date: {domain_info.expiration_date}")
        print(f"Last Updated Date: {domain_info.updated_date}")

        # Check if there are multiple entries (e.g., for a subdomain)
        if isinstance(domain_info.domain_name, list):
            print(f"Multiple domain names found: {domain_info.domain_name}")

        # Display additional information as desired
        print(f"WHOIS Server: {domain_info.whois_server}")
        print(f"Name Servers: {domain_info.name_servers}")
        print(f"Registrant: {domain_info.registrant_name}")

    else:
        print(f"No WHOIS information found for domain: {domain}")

    # Combine the SSL check after retrieving WHOIS information
    url = f"https://{domain}"
    check_ssl_certificate(url)


# Test the function with a domain name
domain = input("Enter a domain name to query WHOIS information: ")
get_whois_info(domain)
