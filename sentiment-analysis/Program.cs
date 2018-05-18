using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;

namespace MlNetSamples.SentimentAnalysis
{
    class Program
    {
        static string _dataPath;
        static string _testDataPath;
        static string _modelPath;
        static void Main(string[] args)
        {
            if(args.Length>0)
            {
                var action = args[0];
                if(action.ToLower()=="train")
                {
                    Console.WriteLine("Enter path to training data file followed by <Enter>:");
                    _dataPath = Console.ReadLine();

                    Console.WriteLine("Enter path to testing data file followed by <Enter>:");
                    _testDataPath = Console.ReadLine();
                   
                   Console.WriteLine("Enter path to model file <Enter>:");
                    _modelPath = Console.ReadLine();
                    
                    //create empty model file if it doesn't already exist
                    if(!File.Exists(_modelPath))
                    {
                        File.CreateText(_modelPath);
                    }

                    //train the model
                    var model = TrainModel();

                    //save the output file
                    model.WriteAsync(_modelPath);

                    Console.WriteLine("Model saved, starting evaluation . . .");
                    
                    //evaluate the model
                    Evaluate(model);
                }
                else if(action.ToLower()=="predict")
                {
                    Console.WriteLine("Enter text to analyze for sentiment followed by <Enter>:");
                    var texttoanalyze = Console.ReadLine();

                    Console.WriteLine("Enter path to model file <Enter>:");
                    _modelPath = Console.ReadLine();

                    //open the model file and instantiate the model
                    var model = PredictionModel.ReadAsync<SentimentData, SentimentPrediction>(_modelPath).Result;

                    //run the prediction
                    Predict(model, texttoanalyze);
                }
                else
                {
                    throw new Exception("Must supply 'train' or 'predict' argument.");
                }
            }
            else
            {
                throw new Exception("Must supply 'train' or 'predict' argument.");
            }
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model, string texttoanalyze)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = texttoanalyze,
                    Sentiment = 0
                }
            };

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Prediction");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => new { sentiment, prediction });

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainModel()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            return model;
        }
    }

    public class SentimentData
    {
        [Column(ordinal: "0")]
        public string SentimentText;
        [Column(ordinal: "1", name: "Label")]
        public float Sentiment;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}
