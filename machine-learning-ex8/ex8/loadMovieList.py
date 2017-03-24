def load_movie_list():
    movie_list = []
    with open('movie_ids.txt') as f:
        lines = f.readlines()
        for line in lines:
            idx, *movie_name = line.split(' ')
            movie_list.append(' '.join(movie_name).rstrip())

    return movie_list
