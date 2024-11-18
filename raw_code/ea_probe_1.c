static int ea_probe(AVProbeData *p)

{

    switch (AV_RL32(&p->buf[0])) {

    case ISNh_TAG:

    case SCHl_TAG:

    case SEAD_TAG:

    case SHEN_TAG:

    case kVGT_TAG:

    case MADk_TAG:

    case MPCh_TAG:

    case MVhd_TAG:

    case MVIh_TAG:

        break;

    default:

        return 0;

    }

    if (AV_RL32(&p->buf[4]) > 0xfffff && AV_RB32(&p->buf[4]) > 0xfffff)

        return 0;



    return AVPROBE_SCORE_MAX;

}
