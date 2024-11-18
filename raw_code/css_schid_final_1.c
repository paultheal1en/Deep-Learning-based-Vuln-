bool css_schid_final(int m, uint8_t cssid, uint8_t ssid, uint16_t schid)

{

    SubchSet *set;

    uint8_t real_cssid;



    real_cssid = (!m && (cssid == 0)) ? channel_subsys.default_cssid : cssid;

    if (real_cssid > MAX_CSSID || ssid > MAX_SSID ||

        !channel_subsys.css[real_cssid] ||

        !channel_subsys.css[real_cssid]->sch_set[ssid]) {

        return true;

    }

    set = channel_subsys.css[real_cssid]->sch_set[ssid];

    return schid > find_last_bit(set->schids_used,

                                 (MAX_SCHID + 1) / sizeof(unsigned long));

}
