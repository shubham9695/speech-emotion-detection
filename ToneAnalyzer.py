from watson_developer_cloud import ToneAnalyzerV3
import json


#function to analyze general tone
#takes json file as input
def detect_emotion(file_name):
    tone_analyzer = ToneAnalyzerV3(username= "1413f188-9420-4639-a1d7-1bc025b428f8",password="MfKDxxTAdpvm",version='2017-09-21')

    #tone(tone_input, content_type='application/json', sentences=None, tones=None,content_language=None, accept_language=None)

    with open(file_name) as tone_json:
        p=tone_json.read()
        #print(p)
        tone = tone_analyzer.tone(p, content_type='application/json', sentences=False, tones=None,content_language=None, accept_language=None)
    with open('transcript2_result.json', 'w') as fp:
        json.dump(tone, fp, indent=2)
    return

#detect_emotion('filname')


"""
#function to analyze customer engagement tone
#takes json file as input
def detect_customer_emotion(file_name):
    tone_analyzer = ToneAnalyzerV3(username= "1413f188-9420-4639-a1d7-1bc025b428f8",password="MfKDxxTAdpvm",version='2017-09-21')

    #tone(tone_input, content_type='application/json', sentences=None, tones=None,content_language=None, accept_language=None)

    with open(file_name) as tone_json:
        tone = tone_analyzer.tone_chat(json.load(tone_json)['utterances'])
    with open('transcript3_result.json', 'w') as fp1:
        json.dump(tone, fp1, indent=2)
    return

#detect_customer_emotion('filename')

"""
