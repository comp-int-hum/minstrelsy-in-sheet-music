Build system that takes a zipped file of material + a jsonl file of metadata from the Lester S. Levy Sheet Music Collection, puts them through the Tesseract OCR engine (via pytesseract) and then trains, applies, and analyzes a variety of gensim topic models over the resulting data. 

Current issues: 1) a large (albeit currently unspecified # of words come back without a topic) 
2) questions of better OCR techniques that need to be thought through. 
