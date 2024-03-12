from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")




dict ={
}

def imgConvertor(url,label):
    image = Image.open(url)
    out = processor( images=image, return_tensors="pt", padding=True)
    dict[label]=out

def similartiyChecker(keyword):
    firstRank=0
    firstRankVal=0

    secondRankVal=0

    thirdRankVal=0

    out = processor(text=[keyword], return_tensors="pt", padding=True)
    for i in dict:  
        
        temp = out.copy()
        temp.update(dict[i])
        outputs = model(**temp)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        print("i: ",probs.item())
        print(probs[0])
        if(probs[0]>firstRank):

            thirdRankVal = secondRankVal
            secondRankVal=firstRankVal

            firstRankVal=i
            firstRank=probs
        
        return firstRank,firstRankVal,secondRankVal,thirdRankVal
    

imgConvertor('deer.jpg','deerImage')
imgConvertor('bird.jpg','birdImage')

print(similartiyChecker('a bird image'))
    