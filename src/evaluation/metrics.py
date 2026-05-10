def calculate_latency(start_time, end_time):

    return round(end_time - start_time, 4)


def calculate_response_length(response):

    return len(response.split())
