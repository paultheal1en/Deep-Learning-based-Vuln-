uint8_t *ff_stream_new_side_data(AVStream *st, enum AVPacketSideDataType type,

                                 int size)

{

    AVPacketSideData *sd, *tmp;

    int i;

    uint8_t *data = av_malloc(size);



    if (!data)

        return NULL;



    for (i = 0; i < st->nb_side_data; i++) {

        sd = &st->side_data[i];



        if (sd->type == type) {

            av_freep(&sd->data);

            sd->data = data;

            sd->size = size;

            return sd->data;

        }

    }



    tmp = av_realloc_array(st->side_data, st->nb_side_data + 1, sizeof(*tmp));

    if (!tmp) {

        av_freep(&data);

        return NULL;

    }



    st->side_data = tmp;

    st->nb_side_data++;



    sd = &st->side_data[st->nb_side_data - 1];

    sd->type = type;

    sd->data = data;

    sd->size = size;

    return data;

}
