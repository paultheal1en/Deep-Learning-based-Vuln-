int ff_h264_get_profile(SPS *sps)

{

    int profile = sps->profile_idc;



    switch (sps->profile_idc) {

    case FF_PROFILE_H264_BASELINE:

        // constraint_set1_flag set to 1

        profile |= (sps->constraint_set_flags & 1 << 1) ? FF_PROFILE_H264_CONSTRAINED : 0;

        break;

    case FF_PROFILE_H264_HIGH_10:

    case FF_PROFILE_H264_HIGH_422:

    case FF_PROFILE_H264_HIGH_444_PREDICTIVE:

        // constraint_set3_flag set to 1

        profile |= (sps->constraint_set_flags & 1 << 3) ? FF_PROFILE_H264_INTRA : 0;

        break;

    }



    return profile;

}
