static int colo_packet_compare_all(Packet *spkt, Packet *ppkt)

{

    trace_colo_compare_main("compare all");

    return colo_packet_compare(ppkt, spkt);

}
