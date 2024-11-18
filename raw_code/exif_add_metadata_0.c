static int exif_add_metadata(AVCodecContext *avctx, int count, int type,

                             const char *name, const char *sep,

                             GetByteContext *gb, int le,

                             AVDictionary **metadata)

{

    switch(type) {

    case 0:

        av_log(avctx, AV_LOG_WARNING,

               "Invalid TIFF tag type 0 found for %s with size %d\n",

               name, count);

        return 0;

    case TIFF_DOUBLE   : return ff_tadd_doubles_metadata(count, name, sep, gb, le, metadata);

    case TIFF_SSHORT   : return ff_tadd_shorts_metadata(count, name, sep, gb, le, 1, metadata);

    case TIFF_SHORT    : return ff_tadd_shorts_metadata(count, name, sep, gb, le, 0, metadata);

    case TIFF_SBYTE    : return ff_tadd_bytes_metadata(count, name, sep, gb, le, 1, metadata);

    case TIFF_BYTE     :

    case TIFF_UNDEFINED: return ff_tadd_bytes_metadata(count, name, sep, gb, le, 0, metadata);

    case TIFF_STRING   : return ff_tadd_string_metadata(count, name, gb, le, metadata);

    case TIFF_SRATIONAL:

    case TIFF_RATIONAL : return ff_tadd_rational_metadata(count, name, sep, gb, le, metadata);

    case TIFF_SLONG    :

    case TIFF_LONG     : return ff_tadd_long_metadata(count, name, sep, gb, le, metadata);

    default:

        avpriv_request_sample(avctx, "TIFF tag type (%u)", type);

        return 0;

    };

}
