void net_tx_pkt_build_vheader(struct NetTxPkt *pkt, bool tso_enable,

    bool csum_enable, uint32_t gso_size)

{

    struct tcp_hdr l4hdr;

    assert(pkt);



    /* csum has to be enabled if tso is. */

    assert(csum_enable || !tso_enable);



    pkt->virt_hdr.gso_type = net_tx_pkt_get_gso_type(pkt, tso_enable);



    switch (pkt->virt_hdr.gso_type & ~VIRTIO_NET_HDR_GSO_ECN) {

    case VIRTIO_NET_HDR_GSO_NONE:

        pkt->virt_hdr.hdr_len = 0;

        pkt->virt_hdr.gso_size = 0;

        break;



    case VIRTIO_NET_HDR_GSO_UDP:

        pkt->virt_hdr.gso_size = IP_FRAG_ALIGN_SIZE(gso_size);

        pkt->virt_hdr.hdr_len = pkt->hdr_len + sizeof(struct udp_header);

        break;



    case VIRTIO_NET_HDR_GSO_TCPV4:

    case VIRTIO_NET_HDR_GSO_TCPV6:

        iov_to_buf(&pkt->vec[NET_TX_PKT_PL_START_FRAG], pkt->payload_frags,

                   0, &l4hdr, sizeof(l4hdr));

        pkt->virt_hdr.hdr_len = pkt->hdr_len + l4hdr.th_off * sizeof(uint32_t);

        pkt->virt_hdr.gso_size = IP_FRAG_ALIGN_SIZE(gso_size);

        break;



    default:

        g_assert_not_reached();

    }



    if (csum_enable) {

        switch (pkt->l4proto) {

        case IP_PROTO_TCP:

            pkt->virt_hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;

            pkt->virt_hdr.csum_start = pkt->hdr_len;

            pkt->virt_hdr.csum_offset = offsetof(struct tcp_hdr, th_sum);

            break;

        case IP_PROTO_UDP:

            pkt->virt_hdr.flags = VIRTIO_NET_HDR_F_NEEDS_CSUM;

            pkt->virt_hdr.csum_start = pkt->hdr_len;

            pkt->virt_hdr.csum_offset = offsetof(struct udp_hdr, uh_sum);

            break;

        default:

            break;

        }

    }

}
