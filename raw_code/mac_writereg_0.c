mac_writereg(E1000State *s, int index, uint32_t val)

{

    uint32_t macaddr[2];



    s->mac_reg[index] = val;



    if (index == RA || index == RA + 1) {

        macaddr[0] = cpu_to_le32(s->mac_reg[RA]);

        macaddr[1] = cpu_to_le32(s->mac_reg[RA + 1]);

        qemu_format_nic_info_str(qemu_get_queue(s->nic), (uint8_t *)macaddr);

    }

}
