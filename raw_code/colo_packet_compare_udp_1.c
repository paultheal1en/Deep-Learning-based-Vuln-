static int colo_packet_compare_udp(Packet *spkt, Packet *ppkt)

{

    int ret;

    int network_header_length = ppkt->ip->ip_hl * 4;



    trace_colo_compare_main("compare udp");



    /*

     * Because of ppkt and spkt are both in the same connection,

     * The ppkt's src ip, dst ip, src port, dst port, ip_proto all are

     * same with spkt. In addition, IP header's Identification is a random

     * field, we can handle it in IP fragmentation function later.

     * COLO just concern the response net packet payload from primary guest

     * and secondary guest are same or not, So we ignored all IP header include

     * other field like TOS,TTL,IP Checksum. we only need to compare

     * the ip payload here.

     */

    ret = colo_packet_compare_common(ppkt, spkt,

                                     network_header_length + ETH_HLEN);



    if (ret) {

        trace_colo_compare_udp_miscompare("primary pkt size", ppkt->size);

        trace_colo_compare_udp_miscompare("Secondary pkt size", spkt->size);

        if (trace_event_get_state(TRACE_COLO_COMPARE_MISCOMPARE)) {

            qemu_hexdump((char *)ppkt->data, stderr, "colo-compare pri pkt",

                         ppkt->size);

            qemu_hexdump((char *)spkt->data, stderr, "colo-compare sec pkt",

                         spkt->size);

        }

    }



    return ret;

}
