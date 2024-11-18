static int colo_packet_compare_icmp(Packet *spkt, Packet *ppkt)

{

    trace_colo_compare_main("compare icmp");



    if (colo_packet_compare_common(ppkt, spkt)) {

        trace_colo_compare_icmp_miscompare("primary pkt size",

                                           ppkt->size);

        qemu_hexdump((char *)ppkt->data, stderr, "colo-compare",

                     ppkt->size);

        trace_colo_compare_icmp_miscompare("Secondary pkt size",

                                           spkt->size);

        qemu_hexdump((char *)spkt->data, stderr, "colo-compare",

                     spkt->size);

        return -1;

    } else {

        return 0;

    }

}
