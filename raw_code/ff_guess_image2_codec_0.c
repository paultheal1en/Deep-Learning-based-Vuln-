enum AVCodecID ff_guess_image2_codec(const char *filename)

{

    return av_str2id(img_tags, filename);

}
