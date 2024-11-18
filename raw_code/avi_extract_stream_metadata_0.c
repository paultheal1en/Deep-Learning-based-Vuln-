static int avi_extract_stream_metadata(AVFormatContext *s, AVStream *st)

{

    GetByteContext gb;

    uint8_t *data = st->codecpar->extradata;

    int data_size = st->codecpar->extradata_size;

    int tag, offset;



    if (!data || data_size < 8) {

        return AVERROR_INVALIDDATA;

    }



    bytestream2_init(&gb, data, data_size);



    tag = bytestream2_get_le32(&gb);



    switch (tag) {

    case MKTAG('A', 'V', 'I', 'F'):

        // skip 4 byte padding

        bytestream2_skip(&gb, 4);

        offset = bytestream2_tell(&gb);

        bytestream2_init(&gb, data + offset, data_size - offset);



        // decode EXIF tags from IFD, AVI is always little-endian

        return avpriv_exif_decode_ifd(s, &gb, 1, 0, &st->metadata);

        break;

    case MKTAG('C', 'A', 'S', 'I'):

        avpriv_request_sample(s, "RIFF stream data tag type CASI (%u)", tag);

        break;

    case MKTAG('Z', 'o', 'r', 'a'):

        avpriv_request_sample(s, "RIFF stream data tag type Zora (%u)", tag);

        break;

    default:

        break;

    }



    return 0;

}
