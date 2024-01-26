input_voice = "hindi1.mp3"


# Speech transcribe and conversion to English
from transformers import WhisperProcessor, WhisperForConditionalGeneration
!pip install torch librosa torchaudio
import librosa
import torch
import torchaudio

####### (Step: 1) Speech transcribe and conversion to English
transcribe_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
transcribe_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")


transcribe_forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")
def transcribe(inp, decoder_ids):
    
    audio_input, _ = librosa.load(inp, sr=16000)

    input_features = transcribe_processor(audio_input, sampling_rate=16000,return_tensors="pt").input_features

    predicted_ids = transcribe_model.generate(input_features, forced_decoder_ids=decoder_ids)

    transcription = transcribe_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

# (Step: 2) conversion of hindi input into english
translate_forced_decoder_ids = processor.get_decoder_prompt_ids(language="hindi", task="translation")


# (Step: 3) Generating Response
from transformers import GPT2LMHeadModel, GPT2Tokenizer

response_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
response_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

def generateResponse(text):

    input_ids = response_tokenizer.encode(text, return_tensors='pt')

    with torch.no_grad():
    output = response_model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)

    generated_text = response_tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


# (Step: 4) translating response back to hindi
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
!pip install sentencepiece 

translation_model_name = "facebook/mbart-large-50-many-to-many-mmt"
translation_token = MBart50TokenizerFast.from_pretrained(translation_model_name)
translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_name)


def translate(text):

    translation_token.src_lang = "en_XX"

    encoded_hi = translation_token(text, return_tensors="pt")

    generated_tokens = translation_model.generate(
        **encoded_hi,
        forced_bos_token_id=translation_token.lang_code_to_id["hi_IN"]
    )

    return translation_token.batch_decode(generated_tokens, skip_special_tokens=True)



# (Step: 5) Genrating voice from text
from transformers import VitsModel, AutoTokenizer
from IPython.display import Audio

speech_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
speech_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

def hind_speech(text):
    inputs = speech_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = speech_model(**inputs).waveform

    return output    




# Main function
def processing(inp):

    hindi_inp = transcribe(inp, transcribe_forced_decoder_ids)
    print(f"Your hindi input is {hindi_inp}")

    english_inp = transcribe(inp, translate_forced_decoder_ids)
    print(f"Your translated input is {english_inp}")

    english_response = generateResponse(english_inp)
    print(f"Your Generated English response is {english_response}")

    hindi_response = translate(english_response)
    print(f"Your Generated hindi response is {hindi_response}")

    output = hind_speech(hindi_response)
    
    return Audio(output.numpy(), rate=model.config.sampling_rate)




