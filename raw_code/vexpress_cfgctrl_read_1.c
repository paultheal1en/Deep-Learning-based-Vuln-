static bool vexpress_cfgctrl_read(arm_sysctl_state *s, unsigned int dcc,

                                  unsigned int function, unsigned int site,

                                  unsigned int position, unsigned int device,

                                  uint32_t *val)

{

    /* We don't support anything other than DCC 0, board stack position 0

     * or sites other than motherboard/daughterboard:

     */

    if (dcc != 0 || position != 0 ||

        (site != SYS_CFG_SITE_MB && site != SYS_CFG_SITE_DB1)) {

        goto cfgctrl_unimp;

    }



    switch (function) {

    case SYS_CFG_VOLT:

        if (site == SYS_CFG_SITE_DB1 && device < s->db_num_vsensors) {

            *val = s->db_voltage[device];

            return true;

        }

        if (site == SYS_CFG_SITE_MB && device == 0) {

            /* There is only one motherboard voltage sensor:

             * VIO : 3.3V : bus voltage between mother and daughterboard

             */

            *val = 3300000;

            return true;

        }

        break;

    case SYS_CFG_OSC:

        if (site == SYS_CFG_SITE_MB && device < sizeof(s->mb_clock)) {

            /* motherboard clock */

            *val = s->mb_clock[device];

            return true;

        }

        if (site == SYS_CFG_SITE_DB1 && device < s->db_num_clocks) {

            /* daughterboard clock */

            *val = s->db_clock[device];

            return true;

        }

        break;

    default:

        break;

    }



cfgctrl_unimp:

    qemu_log_mask(LOG_UNIMP,

                  "arm_sysctl: Unimplemented SYS_CFGCTRL read of function "

                  "0x%x DCC 0x%x site 0x%x position 0x%x device 0x%x\n",

                  function, dcc, site, position, device);

    return false;

}
