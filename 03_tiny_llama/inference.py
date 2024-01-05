import torch
from tokenizers import Tokenizer
from model import TinyLlama
import random
import time

PROMPTS = ['SPAQ', 'God', 'nothing', 'Oh', '334', 'probably', 'So', 'Im', '?', "What's", 'my', 'apparently', 'true', 'There', 'a', "it's", 'Bears', 'Hold', 'volatility', 'Europe', 'Blue', 'Flair', 'Would', 'Iv', 'HULK', 'Hulk', '90%', 'bulls', 'Cause', 'Election', 'Morning', 'Thank', 'UK', 'Awesome', 'holy', 'Drop', 'Like', 'Youre', 'or', 'Throw', 'i', 'Just', 'Nasdaq', 'Dearly', 'Toyota?', 'Awww', 'We', 'yea', 'Someone', 'Jesus', 'Should', 'no', 'Weird', 'Or', 'Leave', 'Silly', 'thats', 'AMERICAAAAAA', 'Dont', 'Pro', 'NIO', "I'd", 'Exactly,', 'Puts', 'Ghost', 'Short?', '337', 'Trick,', 'Short', 'Idk', '.....', 'At', 'Is', 'Lol', '&gt;', 'Dumped', 'Stop', 'Ur', 'Uhhh', '80%', 'Why', 'They', 'Shares?', 'Who', 'Buy', 'Wed', 'Holy', 'Covering', "It's", 'Bull', 'Totally', 'VIX', '*THIS', 'Holding', 'Ive', 'Futures', 'China', 'Msft', 'Jpows', '5.9k', '&gt;VXX', 'Chinese', 'GUHd', 'r', 'Bruh', 'Mad', 'ok', 'Vix', "There's", 'Have', '0', 'Because', 'Need', 'This', 'WE', 'Overbought', 'Unstoppable', 'AOC', 'try', 'Live', 'Never', 'Jed', 'Calls?', '350c', 'Wake', 'mate', 'MMs', 'I', 'gonna', 'Welp', 'Sounds', 'Positions', 'Can', 'Hi', 'early', 'Yes,', 'Yeah', 'Okay', 'Obvious', 'Dear', 'Sell', 'research', 'being', 'That', 'Hes', 'lol', "That's", 'Happens', 'Made', 'while', 'buy', 'yes', 'Yup', 'Yes', 'Fair', 'what', 'Question', 'How', 'Panic', 'light', 'No', 'Cruises', 'Possibly.', 'As', 'just', 'stonk', 'Weeklies', 'roses', 'Wait', 'tendies', "Isn't", 'Surprised', 'the', 'Everyone', 'Been', 'lmao', 'SQQQ', 'Nah.', 'All', '\\*THE', 'Good', 'Ban', '*Market', 'ulls', 'Stonk', 'Sort', 'idk', 'On', 'Nio', 'Shopify', 'Sold', 'The', 'Guess', 'Havent', 'Sleep,', 'here', 'Fuck', 'Last', 'AAPL', 'Same', 'US', 'Day', 'My', 'Mr.Market', 'Alright.', 'TIL', 'Thoughts', 'So,', 'Said', 'Market', 'Dammit!', 'Value', 'in-between', 'For', 'Did', 'guys', 'Biden', 'Well', 'maybe', 'Wym?', 'NIO.', 'jerus', 'What', 'Its', 'market', 'soooooooo', 'recession', '#BLOOD', 'Thats', 'Down', "I'm.....", 'nio', 'Stimulus', 'FUTURES', 'NIO,', 'RIP', 'Itsa', 'Shoutout', 'Only', 'Ok,', '483!', 'Take', 'Looks', 'Look', 'r/Politically_NSFW', 'No,', "there's", 'Futes', 'Here', 'Arent', 'repost', 'ayyye', "5'10", 'Whats', 'Dow', 'Daily', 'this', 'SPY', 'they', 'And', 'Gotta', 'Yeah,', 'Imagine', 'Walked', 'Reminder:', 'acb', 'ATH', 'Yall', 'Where', 'Cry', 'Almost', 'GME', 'If', "Don't", 'Wtf?', '!!', 'Literally', 'More', "don't", 'torrent...', 'Inverse', 'Could', 'Thanks', 'Very', 'Priced', 'A', 'Damn', 'Yes.', '&gt;hope', 'Zimbabwe', 'brother,', 'wait', '2', 'It', 'if', 'PYPL', 'New', 'Tripped,', 'Switched', 'Guys', 'WeBull', 'Fuck,', 'Cool', 'still', 'Fight', "I'm", 'Shit', 'Lmao', 'why', 'Burying', 'dibs', 'Will', 'You', 'What?', 'States', 'TEMPTING', '**', 'Hey']

def main():
    model = torch.load("WallStreetBets_model.pt", map_location="cpu")
    model.mode = "eval"  # so kv cache gets activated
    tokenizer = Tokenizer.from_file("tokenizer.json")
    while True:
        prompt = " " + random.choice(PROMPTS).strip()
        prompt = prompt.encode("ascii", "ignore").decode("ascii")  # ignore all non-ascii characters
        prompt = "[BOS]" + prompt
        generate(model, tokenizer, prompt)

def generate(model: TinyLlama, tokenizer: Tokenizer, prompt: str):
    token_ids = tokenizer.encode(prompt).ids
    step = 0
    eos_id = tokenizer.encode("[EOS]").ids[0]
    print('"', end="", flush=True)
    while step < model.config.context_length:
        input_ = torch.Tensor([[token_ids[step]]]).int()
        time.sleep(0.05)
        logits = model.forward(input_, step)
        if step + 1 < len(token_ids):
            next_token_id = token_ids[step + 1]  # still inputting prompt
        else:
            next_token_id = sample(logits)
            token_ids.append(next_token_id)
            if next_token_id == eos_id:
                print('"', flush=True)
                break
        token = tokenizer.decode([next_token_id]).replace("Ġ", " ").replace("Ċ", "<br>")
        if step == 0: token = token.lstrip()
        print(token, end="", flush=True)
        step += 1

def sample(logits: torch.Tensor):
    # TODO: implement better sampling method like top p or beam search
    return torch.argmax(logits)

if __name__ == "__main__":
    main()
