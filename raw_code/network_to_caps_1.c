static void network_to_caps(RDMACapabilities *cap)

{

    cap->version = ntohl(cap->version);

    cap->flags = ntohl(cap->flags);

}
