static int output_configure(AACContext *ac,

                            uint8_t layout_map[MAX_ELEM_ID * 4][3], int tags,

                            enum OCStatus oc_type, int get_new_frame)

{

    AVCodecContext *avctx = ac->avctx;

    int i, channels = 0, ret;

    uint64_t layout = 0;

    uint8_t id_map[TYPE_END][MAX_ELEM_ID] = {{ 0 }};

    uint8_t type_counts[TYPE_END] = { 0 };



    if (ac->oc[1].layout_map != layout_map) {

        memcpy(ac->oc[1].layout_map, layout_map, tags * sizeof(layout_map[0]));

        ac->oc[1].layout_map_tags = tags;


    for (i = 0; i < tags; i++) {

        int type =         layout_map[i][0];

        int id =           layout_map[i][1];

        id_map[type][id] = type_counts[type]++;






    // Try to sniff a reasonable channel order, otherwise output the

    // channels in the order the PCE declared them.

    if (avctx->request_channel_layout != AV_CH_LAYOUT_NATIVE)

        layout = sniff_channel_order(layout_map, tags);

    for (i = 0; i < tags; i++) {

        int type =     layout_map[i][0];

        int id =       layout_map[i][1];

        int iid =      id_map[type][id];

        int position = layout_map[i][2];

        // Allocate or free elements depending on if they are in the

        // current program configuration.

        ret = che_configure(ac, position, type, iid, &channels);

        if (ret < 0)

            return ret;

        ac->tag_che_map[type][id] = ac->che[type][iid];


    if (ac->oc[1].m4ac.ps == 1 && channels == 2) {

        if (layout == AV_CH_FRONT_CENTER) {

            layout = AV_CH_FRONT_LEFT|AV_CH_FRONT_RIGHT;

        } else {

            layout = 0;





    if (layout) avctx->channel_layout = layout;

                            ac->oc[1].channel_layout = layout;

    avctx->channels       = ac->oc[1].channels       = channels;

    ac->oc[1].status = oc_type;



    if (get_new_frame) {

        if ((ret = frame_configure_elements(ac->avctx)) < 0)

            return ret;




    return 0;
