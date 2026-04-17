#!/bin/bash

python main.py -cff "./config/amazon_products.yaml" -m gspell -adp -tt 5 \
-pdp ./output/products-meta-llama/data.pth \
-pgp ./pretrained/products-meta-llama/gcn.pth \
-ppp ./pretrained/products-meta-llama/projector.pth