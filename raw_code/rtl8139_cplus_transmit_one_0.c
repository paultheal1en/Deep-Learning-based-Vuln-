static int rtl8139_cplus_transmit_one(RTL8139State *s)

{

    if (!rtl8139_transmitter_enabled(s))

    {

        DPRINTF("+++ C+ mode: transmitter disabled\n");

        return 0;

    }



    if (!rtl8139_cp_transmitter_enabled(s))

    {

        DPRINTF("+++ C+ mode: C+ transmitter disabled\n");

        return 0 ;

    }



    int descriptor = s->currCPlusTxDesc;



    target_phys_addr_t cplus_tx_ring_desc =

        rtl8139_addr64(s->TxAddr[0], s->TxAddr[1]);



    /* Normal priority ring */

    cplus_tx_ring_desc += 16 * descriptor;



    DPRINTF("+++ C+ mode reading TX descriptor %d from host memory at "

        "%08x0x%08x = 0x"TARGET_FMT_plx"\n", descriptor, s->TxAddr[1],

        s->TxAddr[0], cplus_tx_ring_desc);



    uint32_t val, txdw0,txdw1,txbufLO,txbufHI;



    cpu_physical_memory_read(cplus_tx_ring_desc,    (uint8_t *)&val, 4);

    txdw0 = le32_to_cpu(val);

    cpu_physical_memory_read(cplus_tx_ring_desc+4,  (uint8_t *)&val, 4);

    txdw1 = le32_to_cpu(val);

    cpu_physical_memory_read(cplus_tx_ring_desc+8,  (uint8_t *)&val, 4);

    txbufLO = le32_to_cpu(val);

    cpu_physical_memory_read(cplus_tx_ring_desc+12, (uint8_t *)&val, 4);

    txbufHI = le32_to_cpu(val);



    DPRINTF("+++ C+ mode TX descriptor %d %08x %08x %08x %08x\n", descriptor,

        txdw0, txdw1, txbufLO, txbufHI);



/* w0 ownership flag */

#define CP_TX_OWN (1<<31)

/* w0 end of ring flag */

#define CP_TX_EOR (1<<30)

/* first segment of received packet flag */

#define CP_TX_FS (1<<29)

/* last segment of received packet flag */

#define CP_TX_LS (1<<28)

/* large send packet flag */

#define CP_TX_LGSEN (1<<27)

/* large send MSS mask, bits 16...25 */

#define CP_TC_LGSEN_MSS_MASK ((1 << 12) - 1)



/* IP checksum offload flag */

#define CP_TX_IPCS (1<<18)

/* UDP checksum offload flag */

#define CP_TX_UDPCS (1<<17)

/* TCP checksum offload flag */

#define CP_TX_TCPCS (1<<16)



/* w0 bits 0...15 : buffer size */

#define CP_TX_BUFFER_SIZE (1<<16)

#define CP_TX_BUFFER_SIZE_MASK (CP_TX_BUFFER_SIZE - 1)

/* w1 add tag flag */

#define CP_TX_TAGC (1<<17)

/* w1 bits 0...15 : VLAN tag (big endian) */

#define CP_TX_VLAN_TAG_MASK ((1<<16) - 1)

/* w2 low  32bit of Rx buffer ptr */

/* w3 high 32bit of Rx buffer ptr */



/* set after transmission */

/* FIFO underrun flag */

#define CP_TX_STATUS_UNF (1<<25)

/* transmit error summary flag, valid if set any of three below */

#define CP_TX_STATUS_TES (1<<23)

/* out-of-window collision flag */

#define CP_TX_STATUS_OWC (1<<22)

/* link failure flag */

#define CP_TX_STATUS_LNKF (1<<21)

/* excessive collisions flag */

#define CP_TX_STATUS_EXC (1<<20)



    if (!(txdw0 & CP_TX_OWN))

    {

        DPRINTF("C+ Tx mode : descriptor %d is owned by host\n", descriptor);

        return 0 ;

    }



    DPRINTF("+++ C+ Tx mode : transmitting from descriptor %d\n", descriptor);



    if (txdw0 & CP_TX_FS)

    {

        DPRINTF("+++ C+ Tx mode : descriptor %d is first segment "

            "descriptor\n", descriptor);



        /* reset internal buffer offset */

        s->cplus_txbuffer_offset = 0;

    }



    int txsize = txdw0 & CP_TX_BUFFER_SIZE_MASK;

    target_phys_addr_t tx_addr = rtl8139_addr64(txbufLO, txbufHI);



    /* make sure we have enough space to assemble the packet */

    if (!s->cplus_txbuffer)

    {

        s->cplus_txbuffer_len = CP_TX_BUFFER_SIZE;

        s->cplus_txbuffer = qemu_malloc(s->cplus_txbuffer_len);

        s->cplus_txbuffer_offset = 0;



        DPRINTF("+++ C+ mode transmission buffer allocated space %d\n",

            s->cplus_txbuffer_len);

    }



    while (s->cplus_txbuffer && s->cplus_txbuffer_offset + txsize >= s->cplus_txbuffer_len)

    {

        s->cplus_txbuffer_len += CP_TX_BUFFER_SIZE;

        s->cplus_txbuffer = qemu_realloc(s->cplus_txbuffer, s->cplus_txbuffer_len);



        DPRINTF("+++ C+ mode transmission buffer space changed to %d\n",

            s->cplus_txbuffer_len);

    }



    if (!s->cplus_txbuffer)

    {

        /* out of memory */



        DPRINTF("+++ C+ mode transmiter failed to reallocate %d bytes\n",

            s->cplus_txbuffer_len);



        /* update tally counter */

        ++s->tally_counters.TxERR;

        ++s->tally_counters.TxAbt;



        return 0;

    }



    /* append more data to the packet */



    DPRINTF("+++ C+ mode transmit reading %d bytes from host memory at "

        TARGET_FMT_plx" to offset %d\n", txsize, tx_addr,

        s->cplus_txbuffer_offset);



    cpu_physical_memory_read(tx_addr, s->cplus_txbuffer + s->cplus_txbuffer_offset, txsize);

    s->cplus_txbuffer_offset += txsize;



    /* seek to next Rx descriptor */

    if (txdw0 & CP_TX_EOR)

    {

        s->currCPlusTxDesc = 0;

    }

    else

    {

        ++s->currCPlusTxDesc;

        if (s->currCPlusTxDesc >= 64)

            s->currCPlusTxDesc = 0;

    }



    /* transfer ownership to target */

    txdw0 &= ~CP_RX_OWN;



    /* reset error indicator bits */

    txdw0 &= ~CP_TX_STATUS_UNF;

    txdw0 &= ~CP_TX_STATUS_TES;

    txdw0 &= ~CP_TX_STATUS_OWC;

    txdw0 &= ~CP_TX_STATUS_LNKF;

    txdw0 &= ~CP_TX_STATUS_EXC;



    /* update ring data */

    val = cpu_to_le32(txdw0);

    cpu_physical_memory_write(cplus_tx_ring_desc,    (uint8_t *)&val, 4);



    /* Now decide if descriptor being processed is holding the last segment of packet */

    if (txdw0 & CP_TX_LS)

    {

        uint8_t dot1q_buffer_space[VLAN_HLEN];

        uint16_t *dot1q_buffer;



        DPRINTF("+++ C+ Tx mode : descriptor %d is last segment descriptor\n",

            descriptor);



        /* can transfer fully assembled packet */



        uint8_t *saved_buffer  = s->cplus_txbuffer;

        int      saved_size    = s->cplus_txbuffer_offset;

        int      saved_buffer_len = s->cplus_txbuffer_len;



        /* create vlan tag */

        if (txdw1 & CP_TX_TAGC) {

            /* the vlan tag is in BE byte order in the descriptor

             * BE + le_to_cpu() + ~swap()~ = cpu */

            DPRINTF("+++ C+ Tx mode : inserting vlan tag with ""tci: %u\n",

                bswap16(txdw1 & CP_TX_VLAN_TAG_MASK));



            dot1q_buffer = (uint16_t *) dot1q_buffer_space;

            dot1q_buffer[0] = cpu_to_be16(ETH_P_8021Q);

            /* BE + le_to_cpu() + ~cpu_to_le()~ = BE */

            dot1q_buffer[1] = cpu_to_le16(txdw1 & CP_TX_VLAN_TAG_MASK);

        } else {

            dot1q_buffer = NULL;

        }



        /* reset the card space to protect from recursive call */

        s->cplus_txbuffer = NULL;

        s->cplus_txbuffer_offset = 0;

        s->cplus_txbuffer_len = 0;



        if (txdw0 & (CP_TX_IPCS | CP_TX_UDPCS | CP_TX_TCPCS | CP_TX_LGSEN))

        {

            DPRINTF("+++ C+ mode offloaded task checksum\n");



            /* ip packet header */

            ip_header *ip = NULL;

            int hlen = 0;

            uint8_t  ip_protocol = 0;

            uint16_t ip_data_len = 0;



            uint8_t *eth_payload_data = NULL;

            size_t   eth_payload_len  = 0;



            int proto = be16_to_cpu(*(uint16_t *)(saved_buffer + 12));

            if (proto == ETH_P_IP)

            {

                DPRINTF("+++ C+ mode has IP packet\n");



                /* not aligned */

                eth_payload_data = saved_buffer + ETH_HLEN;

                eth_payload_len  = saved_size   - ETH_HLEN;



                ip = (ip_header*)eth_payload_data;



                if (IP_HEADER_VERSION(ip) != IP_HEADER_VERSION_4) {

                    DPRINTF("+++ C+ mode packet has bad IP version %d "

                        "expected %d\n", IP_HEADER_VERSION(ip),

                        IP_HEADER_VERSION_4);

                    ip = NULL;

                } else {

                    hlen = IP_HEADER_LENGTH(ip);

                    ip_protocol = ip->ip_p;

                    ip_data_len = be16_to_cpu(ip->ip_len) - hlen;

                }

            }



            if (ip)

            {

                if (txdw0 & CP_TX_IPCS)

                {

                    DPRINTF("+++ C+ mode need IP checksum\n");



                    if (hlen<sizeof(ip_header) || hlen>eth_payload_len) {/* min header length */

                        /* bad packet header len */

                        /* or packet too short */

                    }

                    else

                    {

                        ip->ip_sum = 0;

                        ip->ip_sum = ip_checksum(ip, hlen);

                        DPRINTF("+++ C+ mode IP header len=%d checksum=%04x\n",

                            hlen, ip->ip_sum);

                    }

                }



                if ((txdw0 & CP_TX_LGSEN) && ip_protocol == IP_PROTO_TCP)

                {

#if defined (DEBUG_RTL8139)

                    int large_send_mss = (txdw0 >> 16) & CP_TC_LGSEN_MSS_MASK;

#endif

                    DPRINTF("+++ C+ mode offloaded task TSO MTU=%d IP data %d "

                        "frame data %d specified MSS=%d\n", ETH_MTU,

                        ip_data_len, saved_size - ETH_HLEN, large_send_mss);



                    int tcp_send_offset = 0;

                    int send_count = 0;



                    /* maximum IP header length is 60 bytes */

                    uint8_t saved_ip_header[60];



                    /* save IP header template; data area is used in tcp checksum calculation */

                    memcpy(saved_ip_header, eth_payload_data, hlen);



                    /* a placeholder for checksum calculation routine in tcp case */

                    uint8_t *data_to_checksum     = eth_payload_data + hlen - 12;

                    //                    size_t   data_to_checksum_len = eth_payload_len  - hlen + 12;



                    /* pointer to TCP header */

                    tcp_header *p_tcp_hdr = (tcp_header*)(eth_payload_data + hlen);



                    int tcp_hlen = TCP_HEADER_DATA_OFFSET(p_tcp_hdr);



                    /* ETH_MTU = ip header len + tcp header len + payload */

                    int tcp_data_len = ip_data_len - tcp_hlen;

                    int tcp_chunk_size = ETH_MTU - hlen - tcp_hlen;



                    DPRINTF("+++ C+ mode TSO IP data len %d TCP hlen %d TCP "

                        "data len %d TCP chunk size %d\n", ip_data_len,

                        tcp_hlen, tcp_data_len, tcp_chunk_size);



                    /* note the cycle below overwrites IP header data,

                       but restores it from saved_ip_header before sending packet */



                    int is_last_frame = 0;



                    for (tcp_send_offset = 0; tcp_send_offset < tcp_data_len; tcp_send_offset += tcp_chunk_size)

                    {

                        uint16_t chunk_size = tcp_chunk_size;



                        /* check if this is the last frame */

                        if (tcp_send_offset + tcp_chunk_size >= tcp_data_len)

                        {

                            is_last_frame = 1;

                            chunk_size = tcp_data_len - tcp_send_offset;

                        }



                        DPRINTF("+++ C+ mode TSO TCP seqno %08x\n",

                            be32_to_cpu(p_tcp_hdr->th_seq));



                        /* add 4 TCP pseudoheader fields */

                        /* copy IP source and destination fields */

                        memcpy(data_to_checksum, saved_ip_header + 12, 8);



                        DPRINTF("+++ C+ mode TSO calculating TCP checksum for "

                            "packet with %d bytes data\n", tcp_hlen +

                            chunk_size);



                        if (tcp_send_offset)

                        {

                            memcpy((uint8_t*)p_tcp_hdr + tcp_hlen, (uint8_t*)p_tcp_hdr + tcp_hlen + tcp_send_offset, chunk_size);

                        }



                        /* keep PUSH and FIN flags only for the last frame */

                        if (!is_last_frame)

                        {

                            TCP_HEADER_CLEAR_FLAGS(p_tcp_hdr, TCP_FLAG_PUSH|TCP_FLAG_FIN);

                        }



                        /* recalculate TCP checksum */

                        ip_pseudo_header *p_tcpip_hdr = (ip_pseudo_header *)data_to_checksum;

                        p_tcpip_hdr->zeros      = 0;

                        p_tcpip_hdr->ip_proto   = IP_PROTO_TCP;

                        p_tcpip_hdr->ip_payload = cpu_to_be16(tcp_hlen + chunk_size);



                        p_tcp_hdr->th_sum = 0;



                        int tcp_checksum = ip_checksum(data_to_checksum, tcp_hlen + chunk_size + 12);

                        DPRINTF("+++ C+ mode TSO TCP checksum %04x\n",

                            tcp_checksum);



                        p_tcp_hdr->th_sum = tcp_checksum;



                        /* restore IP header */

                        memcpy(eth_payload_data, saved_ip_header, hlen);



                        /* set IP data length and recalculate IP checksum */

                        ip->ip_len = cpu_to_be16(hlen + tcp_hlen + chunk_size);



                        /* increment IP id for subsequent frames */

                        ip->ip_id = cpu_to_be16(tcp_send_offset/tcp_chunk_size + be16_to_cpu(ip->ip_id));



                        ip->ip_sum = 0;

                        ip->ip_sum = ip_checksum(eth_payload_data, hlen);

                        DPRINTF("+++ C+ mode TSO IP header len=%d "

                            "checksum=%04x\n", hlen, ip->ip_sum);



                        int tso_send_size = ETH_HLEN + hlen + tcp_hlen + chunk_size;

                        DPRINTF("+++ C+ mode TSO transferring packet size "

                            "%d\n", tso_send_size);

                        rtl8139_transfer_frame(s, saved_buffer, tso_send_size,

                            0, (uint8_t *) dot1q_buffer);



                        /* add transferred count to TCP sequence number */

                        p_tcp_hdr->th_seq = cpu_to_be32(chunk_size + be32_to_cpu(p_tcp_hdr->th_seq));

                        ++send_count;

                    }



                    /* Stop sending this frame */

                    saved_size = 0;

                }

                else if (txdw0 & (CP_TX_TCPCS|CP_TX_UDPCS))

                {

                    DPRINTF("+++ C+ mode need TCP or UDP checksum\n");



                    /* maximum IP header length is 60 bytes */

                    uint8_t saved_ip_header[60];

                    memcpy(saved_ip_header, eth_payload_data, hlen);



                    uint8_t *data_to_checksum     = eth_payload_data + hlen - 12;

                    //                    size_t   data_to_checksum_len = eth_payload_len  - hlen + 12;



                    /* add 4 TCP pseudoheader fields */

                    /* copy IP source and destination fields */

                    memcpy(data_to_checksum, saved_ip_header + 12, 8);



                    if ((txdw0 & CP_TX_TCPCS) && ip_protocol == IP_PROTO_TCP)

                    {

                        DPRINTF("+++ C+ mode calculating TCP checksum for "

                            "packet with %d bytes data\n", ip_data_len);



                        ip_pseudo_header *p_tcpip_hdr = (ip_pseudo_header *)data_to_checksum;

                        p_tcpip_hdr->zeros      = 0;

                        p_tcpip_hdr->ip_proto   = IP_PROTO_TCP;

                        p_tcpip_hdr->ip_payload = cpu_to_be16(ip_data_len);



                        tcp_header* p_tcp_hdr = (tcp_header *) (data_to_checksum+12);



                        p_tcp_hdr->th_sum = 0;



                        int tcp_checksum = ip_checksum(data_to_checksum, ip_data_len + 12);

                        DPRINTF("+++ C+ mode TCP checksum %04x\n",

                            tcp_checksum);



                        p_tcp_hdr->th_sum = tcp_checksum;

                    }

                    else if ((txdw0 & CP_TX_UDPCS) && ip_protocol == IP_PROTO_UDP)

                    {

                        DPRINTF("+++ C+ mode calculating UDP checksum for "

                            "packet with %d bytes data\n", ip_data_len);



                        ip_pseudo_header *p_udpip_hdr = (ip_pseudo_header *)data_to_checksum;

                        p_udpip_hdr->zeros      = 0;

                        p_udpip_hdr->ip_proto   = IP_PROTO_UDP;

                        p_udpip_hdr->ip_payload = cpu_to_be16(ip_data_len);



                        udp_header *p_udp_hdr = (udp_header *) (data_to_checksum+12);



                        p_udp_hdr->uh_sum = 0;



                        int udp_checksum = ip_checksum(data_to_checksum, ip_data_len + 12);

                        DPRINTF("+++ C+ mode UDP checksum %04x\n",

                            udp_checksum);



                        p_udp_hdr->uh_sum = udp_checksum;

                    }



                    /* restore IP header */

                    memcpy(eth_payload_data, saved_ip_header, hlen);

                }

            }

        }



        /* update tally counter */

        ++s->tally_counters.TxOk;



        DPRINTF("+++ C+ mode transmitting %d bytes packet\n", saved_size);



        rtl8139_transfer_frame(s, saved_buffer, saved_size, 1,

            (uint8_t *) dot1q_buffer);



        /* restore card space if there was no recursion and reset offset */

        if (!s->cplus_txbuffer)

        {

            s->cplus_txbuffer        = saved_buffer;

            s->cplus_txbuffer_len    = saved_buffer_len;

            s->cplus_txbuffer_offset = 0;

        }

        else

        {

            qemu_free(saved_buffer);

        }

    }

    else

    {

        DPRINTF("+++ C+ mode transmission continue to next descriptor\n");

    }



    return 1;

}
