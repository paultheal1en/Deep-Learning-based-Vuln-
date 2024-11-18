static int colo_packet_compare_tcp(Packet *spkt, Packet *ppkt)

{

    struct tcphdr *ptcp, *stcp;

    int res;



    trace_colo_compare_main("compare tcp");



    ptcp = (struct tcphdr *)ppkt->transport_header;

    stcp = (struct tcphdr *)spkt->transport_header;



    /*

     * The 'identification' field in the IP header is *very* random

     * it almost never matches.  Fudge this by ignoring differences in

     * unfragmented packets; they'll normally sort themselves out if different

     * anyway, and it should recover at the TCP level.

     * An alternative would be to get both the primary and secondary to rewrite

     * somehow; but that would need some sync traffic to sync the state

     */

    if (ntohs(ppkt->ip->ip_off) & IP_DF) {

        spkt->ip->ip_id = ppkt->ip->ip_id;

        /* and the sum will be different if the IDs were different */

        spkt->ip->ip_sum = ppkt->ip->ip_sum;

    }



    /*

     * Check tcp header length for tcp option field.

     * th_off > 5 means this tcp packet have options field.

     * The tcp options maybe always different.

     * for example:

     * From RFC 7323.

     * TCP Timestamps option (TSopt):

     * Kind: 8

     *

     * Length: 10 bytes

     *

     *    +-------+-------+---------------------+---------------------+

     *    |Kind=8 |  10   |   TS Value (TSval)  |TS Echo Reply (TSecr)|

     *    +-------+-------+---------------------+---------------------+

     *       1       1              4                     4

     *

     * In this case the primary guest's timestamp always different with

     * the secondary guest's timestamp. COLO just focus on payload,

     * so we just need skip this field.

     */

    if (ptcp->th_off > 5) {

        ptrdiff_t tcp_offset;



        tcp_offset = ppkt->transport_header - (uint8_t *)ppkt->data

                     + (ptcp->th_off * 4) - ppkt->vnet_hdr_len;

        res = colo_packet_compare_common(ppkt, spkt, tcp_offset);

    } else if (ptcp->th_sum == stcp->th_sum) {

        res = colo_packet_compare_common(ppkt, spkt, ETH_HLEN);

    } else {

        res = -1;

    }



    if (res != 0 && trace_event_get_state(TRACE_COLO_COMPARE_MISCOMPARE)) {

        char pri_ip_src[20], pri_ip_dst[20], sec_ip_src[20], sec_ip_dst[20];



        strcpy(pri_ip_src, inet_ntoa(ppkt->ip->ip_src));

        strcpy(pri_ip_dst, inet_ntoa(ppkt->ip->ip_dst));

        strcpy(sec_ip_src, inet_ntoa(spkt->ip->ip_src));

        strcpy(sec_ip_dst, inet_ntoa(spkt->ip->ip_dst));



        trace_colo_compare_ip_info(ppkt->size, pri_ip_src,

                                   pri_ip_dst, spkt->size,

                                   sec_ip_src, sec_ip_dst);



        trace_colo_compare_tcp_info("pri tcp packet",

                                    ntohl(ptcp->th_seq),

                                    ntohl(ptcp->th_ack),

                                    res, ptcp->th_flags,

                                    ppkt->size);



        trace_colo_compare_tcp_info("sec tcp packet",

                                    ntohl(stcp->th_seq),

                                    ntohl(stcp->th_ack),

                                    res, stcp->th_flags,

                                    spkt->size);



        qemu_hexdump((char *)ppkt->data, stderr,

                     "colo-compare ppkt", ppkt->size);

        qemu_hexdump((char *)spkt->data, stderr,

                     "colo-compare spkt", spkt->size);

    }



    return res;

}
