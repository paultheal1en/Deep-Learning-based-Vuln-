static int image_probe(AVProbeData *p)

{

    if (av_str2id(img_tags, p->filename)) {

        if (av_filename_number_test(p->filename))

            return AVPROBE_SCORE_MAX;

        else

            return AVPROBE_SCORE_MAX/2;

    }

    return 0;

}
