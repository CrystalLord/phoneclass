import transparser
import ann_converter
import consts as ct

line_list = transparser.parse_transcript(ct.TEST_TRANSCRIPT_FP)
line_list = line_list[0:100]

ipa_list = ann_converter.convert(line_list)
cats, converted = ann_converter.category_convert(line_list)
print(cats)
for i in range(len(converted)):
    print(ipa_list[i])
    print(converted[i])
