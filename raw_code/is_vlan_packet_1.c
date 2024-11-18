is_vlan_packet(E1000State *s, const uint8_t *buf)

{

    return (be16_to_cpup((uint16_t *)(buf + 12)) ==

                le16_to_cpup((uint16_t *)(s->mac_reg + VET)));

}
