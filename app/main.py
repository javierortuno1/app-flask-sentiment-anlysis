import logging
import pathlib

import torch
from flask import Flask, render_template, request
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from model import SentimentAnalysis

# Move the globals outside
VOCAB = None
MODEL = None
NGRAMS = None
TOKENIZER = None
MAP_TOKEN2IDX = None

def create_app():
    app = Flask(__name__)
    
    # Load the model and all required components
    def load_model():
        try:
            print("Loading model for the first time!")
            checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
            print(f"Looking for checkpoint at: {checkpoint_path}")
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
                
            checkpoint = torch.load(checkpoint_path)
            print("Checkpoint loaded successfully")

            global VOCAB, MODEL, NGRAMS, TOKENIZER, MAP_TOKEN2IDX
            VOCAB = checkpoint["vocab"]
            print(f"Vocab loaded, size: {len(VOCAB)}")
            
            MODEL = SentimentAnalysis(
                vocab_size=len(VOCAB), 
                embed_dim=checkpoint["embed_dim"], 
                num_class=checkpoint["num_class"]
            )
            MODEL.load_state_dict(checkpoint["model_state_dict"])
            MODEL.eval()
            print("Model loaded and set to eval mode")

            NGRAMS = checkpoint["ngrams"]
            TOKENIZER = get_tokenizer("basic_english")
            MAP_TOKEN2IDX = VOCAB.get_stoi()
            print(f"Tokenizer and vocab mapping initialized. NGRAMS: {NGRAMS}")
            
            print("Model loading completed successfully")
            
        except Exception as e:
            print(f"Error during model loading: {str(e)}")
            raise

    # Load model immediately
    load_model()
    
    if TOKENIZER is None:
        raise RuntimeError("Tokenizer not initialized properly")

    # Disable gradients
    @torch.no_grad()
    def predict_review_sentiment(text):
        if TOKENIZER is None or MAP_TOKEN2IDX is None or MODEL is None:
            raise RuntimeError("Model components not properly initialized")
            
        # Convert text to tensor
        # Changed variable name to be consistent
        text = torch.tensor(
            [MAP_TOKEN2IDX[token] for token in ngrams_iterator(TOKENIZER(text), NGRAMS)]
        )

        # For single text prediction, we don't need offsets
        # We just need to add batch dimension
        text = text.unsqueeze(0)  # Add batch dimension

        # Pass None for offsets when input is 2D (batch_size x sequence_length)
        output = MODEL(text, None)  # Pass None for offsets when input is 2D
        confidences = torch.softmax(output, dim=1)
        return confidences.squeeze()[1].item()

    @app.route("/predict", methods=["POST"])
    def predict():
        """The input parameter is `review`"""
        review = request.form["review"]
        print(f"Prediction for review:\n {review}")
        
        try:
            result = predict_review_sentiment(review)
            return render_template("result.html", result=result)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template("error.html", error=str(e))

    @app.route("/", methods=["GET"])
    def hello():
        """ Return an HTML. """
        return render_template("hello.html")

    @app.errorhandler(500)
    def server_error(e):
        logging.exception('An error occurred during a request.')
        return """
        An internal error occurred: <pre>{}</pre>
        See logs for full stacktrace.
        """.format(e), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)