//package org.apache.spark.example;
package edu.sjsu.cs286; 

import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
import org.apache.spark.storage.StorageLevel;


import scala.Tuple2;

import java.util.*;

public class Recommendation {
    public static final String LINE_SEPERATOR = "::";
    public static final String RESOURCE_PATH = "/mapr/mycluster/user/user01/spark/recommendation/input/"; //absoulute path to the directory that the data files are stored
    public static final String RATINGS_FILE_NAME = "ratings.dat";
    public static final String MOVIES_FILE_NAME = "movies.dat";
    public static final String APP_NAME = "MovieRecommendation";
    public static final String CLUSTER = "local";
    private static JavaSparkContext sc;

    public static void main(String[] args) {
    	Logger.getLogger("org").setLevel(Level.ERROR);
    	
        //Initializing Spark
        SparkConf conf = new SparkConf().setAppName(APP_NAME).setMaster(CLUSTER);
        sc = new JavaSparkContext(conf);

        //Reading external data
        final JavaRDD<String> ratingData = sc.textFile(RESOURCE_PATH + RATINGS_FILE_NAME);
        JavaRDD<String> productData = sc.textFile(RESOURCE_PATH + MOVIES_FILE_NAME);

        System.out.println("Reading input files:");
        System.out.println("Reading "+ RATINGS_FILE_NAME);
        JavaRDD<Tuple2<Integer, Rating>> ratings = ratingData.map(
                new Function<String, Tuple2<Integer, Rating>>() {
                    public Tuple2<Integer, Rating> call(String s) throws Exception {
                        String[] row = s.split(LINE_SEPERATOR);
                        Integer cacheStamp = Integer.parseInt(row[3]) % 10;
                        Rating rating = new Rating(Integer.parseInt(row[0]), Integer.parseInt(row[1]), Double.parseDouble(row[2]));
                        return new Tuple2<Integer, Rating>(cacheStamp, rating);
                    }
                }
        );
        System.out.println("Reading "+ MOVIES_FILE_NAME);
        Map<Integer, String> products = productData.mapToPair(
                new PairFunction<String, Integer, String>() {
                    public Tuple2<Integer, String> call(String s) throws Exception {
                        String[] sarray = s.split(LINE_SEPERATOR);
                        String movieGenre = sarray[1]+LINE_SEPERATOR+sarray[2];
                        return new Tuple2<Integer, String>(Integer.parseInt(sarray[0]),movieGenre);
                    }
                }
        ).collectAsMap();
        
        System.out.println();
        System.out.println("Created JavaRDD for ratings and movies.");
        long ratingCount = ratings.count();
        long userCount = ratings.map(
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user();
                    }
                }
        ).distinct().count();

        long movieCount = ratings.map(
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().product();
                    }
                }
        ).distinct().count();

        System.out.println("Got " + ratingCount + " ratings from "
                + userCount + " users on " + movieCount + " products.");

        System.out.println();
        System.out.println("Splitting ratings data:");
        //Splitting training data
        int numPartitions = 10;
        System.out.println("1. Creating training data split");
        //training data set
        JavaRDD<Rating> training = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() < 6;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).repartition(numPartitions).cache();

        StorageLevel storageLevel = new StorageLevel();
        System.out.println("2. Creating validation data split");
        //validation data set
        JavaRDD<Rating> validation = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 6 && tuple._1() < 8;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).repartition(numPartitions).persist(storageLevel);
        System.out.println("3. Creating testing data split");
        //test data set
        JavaRDD<Rating> test = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 8;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        ).persist(storageLevel);

        long numTraining = training.count();
        long numValidation = validation.count();
        long numTest = test.count();

        System.out.println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest);
        System.out.println();
        System.out.println("Training the training data using ALS.train");
        //training model
        
        //Due to time constraint, we will test only 8 combinations resulting 
        //from the cross product of 2 different ranks (8 and 12), 2 different lambdas 
        //(1.0 and 10.0), and two different numbers of iterations (10 and 20).
        int[] ranks = {8, 12};            //number of latent factors in als
        float[] lambdas = {0.1f, 10.0f}; //regularizaton pattern in als, 
        int[] numIters = {10, 20};       //number of iterations to run

        double bestValidationRmse = Double.MAX_VALUE;
        int bestRank = 0;
        float bestLambda = -1.0f;
        int bestNumIter = -1;
        MatrixFactorizationModel bestModel = null;
        //training a bunch of models(using training set) and calculating RMSE(using validation set) for each
        //the lowest RMSE is picked as the best model
        for (int currentRank : ranks) {
            for (float currentLambda : lambdas) {
                for (int currentNumIter : numIters) {
                	
                    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(training), currentRank, currentNumIter, currentLambda);

                    Double validationRmse = computeRMSE(model, validation);
                    System.out.println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
                            + currentRank + ", lambda = " + currentLambda + ", and numIter = " + currentNumIter + ".");

                    if (validationRmse < bestValidationRmse) {
                        bestModel = model;
                        bestValidationRmse = validationRmse;
                        bestRank = currentRank;
                        bestLambda = currentLambda;
                        bestNumIter = currentNumIter;
                    }
                }
            }
        }

        //Computing Root Mean Square Error in the test dataset
        Double testRmse = computeRMSE(bestModel, test);
        RDD<Tuple2<Object, double[]>> features = bestModel.productFeatures();
        System.out.println();
        System.out.println("Saving the best model");
        bestModel.save(sc.sc(), "/mapr/mycluster/user/user01/spark/recommendation/output/model");
        features.saveAsTextFile(RESOURCE_PATH + "features");
        System.out.println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda +
                           ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".");
        System.out.println();
        System.out.println("Model creation completed. Moving to prediction phase.");
        System.out.println("Loading the best model");
        bestModel = MatrixFactorizationModel.load(sc.sc(), "/mapr/mycluster/user/user01/spark/recommendation/output/model");
        
        int userId = Integer.parseInt(args[0]);
        System.out.println();
        System.out.println("Getting recommendation for the user:"+userId);
        List<Rating> recommendations = getRecommendations(userId, bestModel, ratings, products);
        //hashmap for genre and list of movies in that genre
        HashMap<String,ArrayList<String>> genreMovieMap = new HashMap<>();
        //Printing Recommendations
        System.out.println("Printing recommendations for user: " + userId);
        int count = 0;
        for (Rating recommendation : recommendations) {
            if (products.containsKey(recommendation.product())) {
            	count++;
            	String movieWithGenre = products.get(recommendation.product());
            	String[] arr = movieWithGenre.split(LINE_SEPERATOR);
            	String genre = arr[1];
            	ArrayList<String> movies = genreMovieMap.get(genre);
            	if(movies == null){
            		ArrayList<String> newMovieList = new ArrayList<>();
            		newMovieList.add(recommendation.product() + " "+arr[0]);
            		genreMovieMap.put(genre,newMovieList);
            	}else{
            		movies.add(recommendation.product() + " "+arr[0]);	
            	}
            	
                System.out.println(recommendation.product() + " " + products.get(recommendation.product()) + " Rating:" + recommendation.rating());
            }
            if(count == 20)
            	break;
        }
        
       
        
        
        HashSet<String> userGenre = getRecommendationsOfUserGenre(userId, bestModel, ratings, products);
        
        System.out.println("Is the userGenre hash set empty:? "+ userGenre.isEmpty());
        HashMap<String,ArrayList<String>> usergGenreMovieMap = new HashMap<>();
        //Printing Recommendations
        System.out.println("Printing recommendations for user's genre: " + userId);
       
        for (Rating recommendation : recommendations) {
            if (products.containsKey(recommendation.product())) {
            	String movieWithGenre = products.get(recommendation.product());
            	String[] arr = movieWithGenre.split(LINE_SEPERATOR);
            	String genre = arr[1];
            	if(userGenre.contains(genre)){
	            	ArrayList<String> movies = usergGenreMovieMap.get(genre);
	            	if(movies == null){
	            		ArrayList<String> newMovieList = new ArrayList<>();
	            		newMovieList.add(arr[0]);
	            		usergGenreMovieMap.put(genre,newMovieList);
	            	}else{
	            		movies.add(arr[0]);	
	            	}
	            	
	                System.out.println(recommendation.product() + " " + products.get(recommendation.product()) + " Rating:" + recommendation.rating());
            	}
            }
        }
        
        System.out.println();
        System.out.println("Printing recommendations for user based on the genre:");
       // System.out.println();
        for(Map.Entry obj: usergGenreMovieMap.entrySet()){
        	System.out.println();
        	System.out.println("Genre recommended :"+obj.getKey());
        	ArrayList<String> movies = (ArrayList<String>)obj.getValue();
        	for(String movie: movies){
        		System.out.println(movie);
        	}
        }

    }

    /**
     * Calculating the Root Mean Squared Error
     *
     * @param model best model generated.
     * @param data  rating data.
     * @return      Root Mean Squared Error
     */
    //references:
    //1. https://spark.apache.org/docs/1.5.2/mllib-collaborative-filtering.html
    //2. https://mahout.apache.org/users/recommender/matrix-factorization.html
    //we scale the regularization parameter lambda in solving each least squares problem by the number of ratings 
    //the user generated in updating user factors, or the number of ratings the product received in 
    //updating product factors. This approach is named “ALS-WR” - weighted lambda regularization
    // It makes lambda less dependent on the scale of the dataset. 
    //So we can apply the best parameter learned from a sampled subset to the full dataset and expect similar performance.
    public static Double computeRMSE(MatrixFactorizationModel model, JavaRDD<Rating> data) {
        JavaRDD<Tuple2<Object, Object>> userProducts = data.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );

        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                        new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                ));
        JavaRDD<Tuple2<Double, Double>> predictionsAndRatings =
                JavaPairRDD.fromJavaRDD(data.map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                        new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                )).join(predictions).values();

        double mse =  JavaDoubleRDD.fromRDD(predictionsAndRatings.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double err = pair._1() - pair._2();
                        return err * err;
                    }
                }
        ).rdd()).mean();

        return Math.sqrt(mse);
    }

    /**
     * Returns the list of recommendations for a given user
     *
     * @param userId    user id.
     * @param model     best model.
     * @param ratings   rating data.
     * @param products  product list.
     * @return          The list of recommended products.
     */
    private static List<Rating> getRecommendations(final int userId, MatrixFactorizationModel model, JavaRDD<Tuple2<Integer, Rating>> ratings, Map<Integer, String> products) {
        List<Rating> recommendations;

        //Getting the users ratings
        JavaRDD<Rating> userRatings = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user() == userId;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        );

        //Getting the product ID's of the products that user rated
        JavaRDD<Tuple2<Object, Object>> userProducts = userRatings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );

        List<Integer> productSet = new ArrayList<Integer>();
        productSet.addAll(products.keySet());

        Iterator<Tuple2<Object, Object>> productIterator = userProducts.toLocalIterator();

        //Removing the user watched (rated) set from the all product set
        while(productIterator.hasNext()) {
            Integer movieId = (Integer)productIterator.next()._2();
            if(productSet.contains(movieId)){
                productSet.remove(movieId);
            }
        }

        JavaRDD<Integer> candidates = sc.parallelize(productSet);

        JavaRDD<Tuple2<Integer, Integer>> userCandidates = candidates.map(
                new Function<Integer, Tuple2<Integer, Integer>>() {
                    public Tuple2<Integer, Integer> call(Integer integer) throws Exception {
                        return new Tuple2<Integer, Integer>(userId, integer);
                    }
                }
        );

        //Predict recommendations for the given user
        recommendations = model.predict(JavaPairRDD.fromJavaRDD(userCandidates)).collect();

        //Sorting the recommended products and sort them according to the rating
//        Collections.sort(recommendations, new Comparator<Rating>() {
//            public int compare(Rating r1, Rating r2) {
//                return r1.rating() < r2.rating() ? -1 : r1.rating() > r2.rating() ? 1 : 0;
//            }
//        });
        
        Collections.sort(recommendations, new Comparator<Rating>() {
            public int compare(Rating r1, Rating r2) {
                return r1.rating() < r2.rating() ? 1 : r1.rating() > r2.rating() ? -1 : 0;
            }
        });

        //get top 50 from the recommended products.
        recommendations = recommendations.subList(0, 50);

        return recommendations;
    }
    
    /**
     * Returns the list of recommendations for a given user
     *
     * @param userId    user id.
     * @param model     best model.
     * @param ratings   rating data.
     * @param products  product list.
     * @return          The list of recommended products.
     */
    private static HashSet<String> getRecommendationsOfUserGenre(final int userId, MatrixFactorizationModel model, JavaRDD<Tuple2<Integer, Rating>> ratings, final Map<Integer, String> products) {
        //List<Rating> recommendations;
        final HashSet<String> genreSet = new HashSet<>();
        //Getting the users ratings
        JavaRDD<Rating> userRatings = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user() == userId;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        );

        //Getting the product ID's of the products that user rated
        JavaRDD<Tuple2<Object, Object>> userProducts = userRatings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                    	String movieGenre = products.get(r.product());
                    	System.out.println("movieGenre="+movieGenre);
                    	String[] movieGenreSplits = movieGenre.split(LINE_SEPERATOR);
                    	genreSet.add(movieGenreSplits[1]);
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        return genreSet;
//        List<Integer> productSet = new ArrayList<Integer>();
//        productSet.addAll(products.keySet());
//
//        Iterator<Tuple2<Object, Object>> productIterator = userProducts.toLocalIterator();
//
//        //Removing the user watched (rated) set from the all product set
//        while(productIterator.hasNext()) {
//            Integer movieId = (Integer)productIterator.next()._2();
//            String movieGenre = products.get(movieId);
//            String[] movieGenreSplits = movieGenre.split(LINE_SEPERATOR);
//            
//            if(productSet.contains(movieId) || !genreSet.contains(movieGenreSplits[1])){
//                productSet.remove(movieId);
//            }
//        }
//
//        JavaRDD<Integer> candidates = sc.parallelize(productSet);
//
//        JavaRDD<Tuple2<Integer, Integer>> userCandidates = candidates.map(
//                new Function<Integer, Tuple2<Integer, Integer>>() {
//                    public Tuple2<Integer, Integer> call(Integer integer) throws Exception {
//                        return new Tuple2<Integer, Integer>(userId, integer);
//                    }
//                }
//        );
//
//        //Predict recommendations for the given user
//        recommendations = model.predict(JavaPairRDD.fromJavaRDD(userCandidates)).collect();
//
//        //Sorting the recommended products and sort them according to the rating
////        Collections.sort(recommendations, new Comparator<Rating>() {
////            public int compare(Rating r1, Rating r2) {
////                return r1.rating() < r2.rating() ? -1 : r1.rating() > r2.rating() ? 1 : 0;
////            }
////        });
//        
//        Collections.sort(recommendations, new Comparator<Rating>() {
//            public int compare(Rating r1, Rating r2) {
//                return r1.rating() < r2.rating() ? 1 : r1.rating() > r2.rating() ? -1 : 0;
//            }
//        });
//
//        //get top 50 from the recommended products.
//        recommendations = recommendations.subList(0, 20);
//
//        return recommendations;
    }
}
