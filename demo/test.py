#%%
from model_utils import preprocess_text

for q in ["telephone", "Find the area of a circle with radius 5"]:
    print("RAW:    ", q)
    print("CLEAN:  ", preprocess_text(q))
    print()

# %%
