static av_cold int hevc_sdp_parse_fmtp_config(AVFormatContext *s,

                                              AVStream *stream,

                                              PayloadContext *hevc_data,

                                              char *attr, char *value)

{

    /* profile-space: 0-3 */

    /* profile-id: 0-31 */

    if (!strcmp(attr, "profile-id")) {

        hevc_data->profile_id = atoi(value);

        av_dlog(s, "SDP: found profile-id: %d\n", hevc_data->profile_id);

    }



    /* tier-flag: 0-1 */

    /* level-id: 0-255 */

    /* interop-constraints: [base16] */

    /* profile-compatibility-indicator: [base16] */

    /* sprop-sub-layer-id: 0-6, defines highest possible value for TID, default: 6 */

    /* recv-sub-layer-id: 0-6 */

    /* max-recv-level-id: 0-255 */

    /* tx-mode: MSM,SSM */

    /* sprop-vps: [base64] */

    /* sprop-sps: [base64] */

    /* sprop-pps: [base64] */

    /* sprop-sei: [base64] */

    if (!strcmp(attr, "sprop-vps") || !strcmp(attr, "sprop-sps") ||

        !strcmp(attr, "sprop-pps") || !strcmp(attr, "sprop-sei")) {

        uint8_t **data_ptr;

        int *size_ptr;

        if (!strcmp(attr, "sprop-vps")) {

            data_ptr = &hevc_data->vps;

            size_ptr = &hevc_data->vps_size;

        } else if (!strcmp(attr, "sprop-sps")) {

            data_ptr = &hevc_data->sps;

            size_ptr = &hevc_data->sps_size;

        } else if (!strcmp(attr, "sprop-pps")) {

            data_ptr = &hevc_data->pps;

            size_ptr = &hevc_data->pps_size;

        } else if (!strcmp(attr, "sprop-sei")) {

            data_ptr = &hevc_data->sei;

            size_ptr = &hevc_data->sei_size;

        }



        while (*value) {

            char base64packet[1024];

            uint8_t decoded_packet[1024];

            int decoded_packet_size;

            char *dst = base64packet;



            while (*value && *value != ',' &&

                   (dst - base64packet) < sizeof(base64packet) - 1) {

                *dst++ = *value++;

            }

            *dst++ = '\0';



            if (*value == ',')

                value++;



            decoded_packet_size = av_base64_decode(decoded_packet, base64packet,

                                                   sizeof(decoded_packet));

            if (decoded_packet_size > 0) {

                uint8_t *tmp = av_realloc(*data_ptr, decoded_packet_size +

                                          sizeof(start_sequence) + *size_ptr);

                if (!tmp) {

                    av_log(s, AV_LOG_ERROR,

                           "Unable to allocate memory for extradata!\n");

                    return AVERROR(ENOMEM);

                }

                *data_ptr = tmp;



                memcpy(*data_ptr + *size_ptr, start_sequence,

                       sizeof(start_sequence));

                memcpy(*data_ptr + *size_ptr + sizeof(start_sequence),

                       decoded_packet, decoded_packet_size);



                *size_ptr += sizeof(start_sequence) + decoded_packet_size;

            }

        }

    }



    /* max-lsr, max-lps, max-cpb, max-dpb, max-br, max-tr, max-tc */

    /* max-fps */



    /* sprop-max-don-diff: 0-32767



         When the RTP stream depends on one or more other RTP

         streams (in this case tx-mode MUST be equal to "MSM" and

         MSM is in use), this parameter MUST be present and the

         value MUST be greater than 0.

    */

    if (!strcmp(attr, "sprop-max-don-diff")) {

        if (atoi(value) > 0)

            hevc_data->using_donl_field = 1;

        av_dlog(s, "Found sprop-max-don-diff in SDP, DON field usage is: %d\n",

                hevc_data->using_donl_field);

    }



    /* sprop-depack-buf-nalus: 0-32767 */

    if (!strcmp(attr, "sprop-depack-buf-nalus")) {

        if (atoi(value) > 0)

            hevc_data->using_donl_field = 1;

        av_dlog(s, "Found sprop-depack-buf-nalus in SDP, DON field usage is: %d\n",

                hevc_data->using_donl_field);

    }



    /* sprop-depack-buf-bytes: 0-4294967295 */

    /* depack-buf-cap */

    /* sprop-segmentation-id: 0-3 */

    /* sprop-spatial-segmentation-idc: [base16] */

    /* dec-parallel-ca: */

    /* include-dph */



    return 0;

}
