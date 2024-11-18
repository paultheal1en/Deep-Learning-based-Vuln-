static int colo_packet_compare_udp(Packet *spkt, Packet *ppkt)

{

    int ret;



    trace_colo_compare_main("compare udp");



    ret = colo_packet_compare_common(ppkt, spkt);



    if (ret) {

        trace_colo_compare_udp_miscompare("primary pkt size", ppkt->size);

        qemu_hexdump((char *)ppkt->data, stderr, "colo-compare", ppkt->size);

        trace_colo_compare_udp_miscompare("Secondary pkt size", spkt->size);

        qemu_hexdump((char *)spkt->data, stderr, "colo-compare", spkt->size);

    }



    return ret;

}
