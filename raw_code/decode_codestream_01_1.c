static int decode_codestream(J2kDecoderContext *s)

{

    J2kCodingStyle *codsty = s->codsty;

    J2kQuantStyle  *qntsty = s->qntsty;

    uint8_t *properties = s->properties;



    for (;;){

        int marker, len, ret = 0;

        const uint8_t *oldbuf;

        if (s->buf_end - s->buf < 2){

            av_log(s->avctx, AV_LOG_ERROR, "Missing EOC\n");

            break;

        }



        marker = bytestream_get_be16(&s->buf);

        if(s->avctx->debug & FF_DEBUG_STARTCODE)

            av_log(s->avctx, AV_LOG_DEBUG, "marker 0x%.4X at pos 0x%tx\n", marker, s->buf - s->buf_start - 4);

        oldbuf = s->buf;



        if (marker == J2K_SOD){

            J2kTile *tile = s->tile + s->curtileno;

            if (ret = init_tile(s, s->curtileno))

                return ret;

            if (ret = decode_packets(s, tile))

                return ret;

            continue;

        }

        if (marker == J2K_EOC)

            break;



        if (s->buf_end - s->buf < 2)

            return AVERROR(EINVAL);

        len = bytestream_get_be16(&s->buf);

        switch(marker){

            case J2K_SIZ:

                ret = get_siz(s); break;

            case J2K_COC:

                ret = get_coc(s, codsty, properties); break;

            case J2K_COD:

                ret = get_cod(s, codsty, properties); break;

            case J2K_QCC:

                ret = get_qcc(s, len, qntsty, properties); break;

            case J2K_QCD:

                ret = get_qcd(s, len, qntsty, properties); break;

            case J2K_SOT:

                if (!(ret = get_sot(s))){

                    codsty = s->tile[s->curtileno].codsty;

                    qntsty = s->tile[s->curtileno].qntsty;

                    properties = s->tile[s->curtileno].properties;

                }

                break;

            case J2K_COM:

                // the comment is ignored

                s->buf += len - 2; break;

            default:

                av_log(s->avctx, AV_LOG_ERROR, "unsupported marker 0x%.4X at pos 0x%tx\n", marker, s->buf - s->buf_start - 4);

                s->buf += len - 2; break;

        }

        if (s->buf - oldbuf != len || ret){

            av_log(s->avctx, AV_LOG_ERROR, "error during processing marker segment %.4x\n", marker);

            return ret ? ret : -1;

        }

    }

    return 0;

}
