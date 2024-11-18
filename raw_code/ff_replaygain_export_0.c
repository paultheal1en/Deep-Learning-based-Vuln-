int ff_replaygain_export(AVStream *st, AVDictionary *metadata)

{

    const AVDictionaryEntry *tg, *tp, *ag, *ap;



    tg = av_dict_get(metadata, "REPLAYGAIN_TRACK_GAIN", NULL, 0);

    tp = av_dict_get(metadata, "REPLAYGAIN_TRACK_PEAK", NULL, 0);

    ag = av_dict_get(metadata, "REPLAYGAIN_ALBUM_GAIN", NULL, 0);

    ap = av_dict_get(metadata, "REPLAYGAIN_ALBUM_PEAK", NULL, 0);



    return replaygain_export(st,

                             tg ? tg->value : NULL,

                             tp ? tp->value : NULL,

                             ag ? ag->value : NULL,

                             ap ? ap->value : NULL);

}
