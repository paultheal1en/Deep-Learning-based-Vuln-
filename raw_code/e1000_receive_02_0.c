e1000_receive(void *opaque, const uint8_t *buf, size_t size)

{

    E1000State *s = opaque;

    struct e1000_rx_desc desc;

    target_phys_addr_t base;

    unsigned int n, rdt;

    uint32_t rdh_start;

    uint16_t vlan_special = 0;

    uint8_t vlan_status = 0, vlan_offset = 0;



    if (!(s->mac_reg[RCTL] & E1000_RCTL_EN))

        return;



    if (size > s->rxbuf_size) {

        DBGOUT(RX, "packet too large for buffers (%lu > %d)\n",

               (unsigned long)size, s->rxbuf_size);

        return;

    }



    if (!receive_filter(s, buf, size))

        return;



    if (vlan_enabled(s) && is_vlan_packet(s, buf)) {

        vlan_special = cpu_to_le16(be16_to_cpup((uint16_t *)(buf + 14)));

        memmove((void *)(buf + 4), buf, 12);

        vlan_status = E1000_RXD_STAT_VP;

        vlan_offset = 4;

        size -= 4;

    }



    rdh_start = s->mac_reg[RDH];

    size += 4; // for the header

    do {

        if (s->mac_reg[RDH] == s->mac_reg[RDT] && s->check_rxov) {

            set_ics(s, 0, E1000_ICS_RXO);

            return;

        }

        base = ((uint64_t)s->mac_reg[RDBAH] << 32) + s->mac_reg[RDBAL] +

               sizeof(desc) * s->mac_reg[RDH];

        cpu_physical_memory_read(base, (void *)&desc, sizeof(desc));

        desc.special = vlan_special;

        desc.status |= (vlan_status | E1000_RXD_STAT_DD);

        if (desc.buffer_addr) {

            cpu_physical_memory_write(le64_to_cpu(desc.buffer_addr),

                                      (void *)(buf + vlan_offset), size);

            desc.length = cpu_to_le16(size);

            desc.status |= E1000_RXD_STAT_EOP|E1000_RXD_STAT_IXSM;

        } else // as per intel docs; skip descriptors with null buf addr

            DBGOUT(RX, "Null RX descriptor!!\n");

        cpu_physical_memory_write(base, (void *)&desc, sizeof(desc));



        if (++s->mac_reg[RDH] * sizeof(desc) >= s->mac_reg[RDLEN])

            s->mac_reg[RDH] = 0;

        s->check_rxov = 1;

        /* see comment in start_xmit; same here */

        if (s->mac_reg[RDH] == rdh_start) {

            DBGOUT(RXERR, "RDH wraparound @%x, RDT %x, RDLEN %x\n",

                   rdh_start, s->mac_reg[RDT], s->mac_reg[RDLEN]);

            set_ics(s, 0, E1000_ICS_RXO);

            return;

        }

    } while (desc.buffer_addr == 0);



    s->mac_reg[GPRC]++;

    s->mac_reg[TPR]++;

    n = s->mac_reg[TORL];

    if ((s->mac_reg[TORL] += size) < n)

        s->mac_reg[TORH]++;



    n = E1000_ICS_RXT0;

    if ((rdt = s->mac_reg[RDT]) < s->mac_reg[RDH])

        rdt += s->mac_reg[RDLEN] / sizeof(desc);

    if (((rdt - s->mac_reg[RDH]) * sizeof(desc)) <= s->mac_reg[RDLEN] >>

        s->rxbuf_min_shift)

        n |= E1000_ICS_RXDMT0;



    set_ics(s, 0, n);

}
