import collections,csv,sys


def process_ratings(ratings,output):

    whole = []
    file = open(ratings)
    for line in file:
        row = str(line).strip().split('::')
        whole.append(row)
    writer = csv.writer(open(output, 'w'))
    # writer.writerow(["movieId", "rating", "userId", "timestamp"])
    writer.writerow(["userId", "movieId", "rating", "timestamp"])

    for row in whole:
        writer.writerow(row)

def write_movie_sents(filename,output):
    """
    :param filename: input rating_data file
    :param output: movie sentence file
    :return:
    """
    print("-Start sorting ratings")

    """
     huge is a dict where key means user id and value is a
     list of movies watched by this user,
     along with timestamp and rating.
    """
    huge = collections.defaultdict(collections.defaultdict)
    # build whole
    # df = pd.read_csv(filename)
    # print ("-Loading in {} records from {}".format(len(df),filename))
    # for i in range(100000):
    #     if (i%10000) == 0:
    #         print ("{} lines of data loaded".format(i))
    #     userId = int(df.loc[i,'userId'])
    #     timeStamp = int(df.loc[i,'timestamp'])
    #     movieId = int(df.loc[i,'movieId'])
    #     rating = int(df.loc[i,'rating'])
    #     data = (movieId, timeStamp, rating)
    #     if userId in huge.keys():
    #         huge[userId].append(data)
    #     else:
    #         huge[userId] = [data]
    with open(filename,'r') as f:
        for i,line in enumerate(f.readlines()):
            if i == 0:
                continue
            if (i%10000) == 0:
                print ("{} lines of data loaded".format(i))
            line = line.split(",")
            userId = int(line[0])
            timeStamp = int(line[3])
            movieId = int(line[1])
            rating = line[2]
            data = (movieId, timeStamp, rating)
            if userId in huge.keys():
                huge[userId].append(data)
            else:
                huge[userId] = [data]

    # sort by timestamp
    for k,v in huge.items():
        huge[k] = sorted(v, key=lambda t: t[1])
    print ("-{} users loaded from {}".format(len(huge),filename))
    print("-Writing movie sentences")
    train = output
    resTrain = open(train, 'w')
    for k, v in huge.items():
        # drop, keep = randomSelect(list(v.values()))
        movies = [str(x[0]) for x in v]
        resTrain.write(",".join(movies)+'\n')
        ratings = [str(x[2]) for x in v]
        assert len(movies) == len(ratings)
        resTrain.write(",".join(ratings)+'\n')

    resTrain.close()
    print('Writting finish!')

def main():
    if len(sys.argv) == 4:
        #10m
        raw_ratings_file = sys.argv[1]
        ratings_data = sys.argv[2]
        movie_sents = sys.argv[3]

        process_ratings(raw_ratings_file,ratings_data)
        write_movie_sents(ratings_data,movie_sents)
    if len(sys.argv) == 3:
        #20m
        ratings_data = sys.argv[1]
        movie_sents = sys.argv[2]
        write_movie_sents(ratings_data,movie_sents)
    return 0

if __name__ == "__main__":
    sys.exit(main())