static int ebml_parse_elem(MatroskaDemuxContext *matroska,

                           EbmlSyntax *syntax, void *data)

{

    static const uint64_t max_lengths[EBML_TYPE_COUNT] = {

        [EBML_UINT]  = 8,

        [EBML_FLOAT] = 8,

        // max. 16 MB for strings

        [EBML_STR]   = 0x1000000,

        [EBML_UTF8]  = 0x1000000,

        // max. 256 MB for binary data

        [EBML_BIN]   = 0x10000000,

        // no limits for anything else

    };

    AVIOContext *pb = matroska->ctx->pb;

    uint32_t id = syntax->id;

    uint64_t length;

    int res;



    data = (char *)data + syntax->data_offset;

    if (syntax->list_elem_size) {

        EbmlList *list = data;

        list->elem = av_realloc(list->elem, (list->nb_elem+1)*syntax->list_elem_size);

        data = (char*)list->elem + list->nb_elem*syntax->list_elem_size;

        memset(data, 0, syntax->list_elem_size);

        list->nb_elem++;

    }



    if (syntax->type != EBML_PASS && syntax->type != EBML_STOP) {

        matroska->current_id = 0;

        if ((res = ebml_read_length(matroska, pb, &length)) < 0)

            return res;

        if (max_lengths[syntax->type] && length > max_lengths[syntax->type]) {

            av_log(matroska->ctx, AV_LOG_ERROR,

                   "Invalid length 0x%"PRIx64" > 0x%"PRIx64" for syntax element %i\n",

                   length, max_lengths[syntax->type], syntax->type);

            return AVERROR_INVALIDDATA;

        }

    }



    switch (syntax->type) {

    case EBML_UINT:  res = ebml_read_uint  (pb, length, data);  break;

    case EBML_FLOAT: res = ebml_read_float (pb, length, data);  break;

    case EBML_STR:

    case EBML_UTF8:  res = ebml_read_ascii (pb, length, data);  break;

    case EBML_BIN:   res = ebml_read_binary(pb, length, data);  break;

    case EBML_NEST:  if ((res=ebml_read_master(matroska, length)) < 0)

                         return res;

                     if (id == MATROSKA_ID_SEGMENT)

                         matroska->segment_start = avio_tell(matroska->ctx->pb);

                     return ebml_parse_nest(matroska, syntax->def.n, data);

    case EBML_PASS:  return ebml_parse_id(matroska, syntax->def.n, id, data);

    case EBML_STOP:  return 1;

    default:         return avio_skip(pb,length)<0 ? AVERROR(EIO) : 0;

    }

    if (res == AVERROR_INVALIDDATA)

        av_log(matroska->ctx, AV_LOG_ERROR, "Invalid element\n");

    else if (res == AVERROR(EIO))

        av_log(matroska->ctx, AV_LOG_ERROR, "Read error\n");

    return res;

}
