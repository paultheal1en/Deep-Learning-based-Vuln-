static void free_packet_list(AVPacketList *pktl)

{

    AVPacketList *cur;

    while (pktl) {

        cur = pktl;

        pktl = cur->next;

        av_free_packet(&cur->pkt);

        av_free(cur);

    }

}
