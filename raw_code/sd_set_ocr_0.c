static void sd_set_ocr(SDState *sd)

{

    /* All voltages OK, card power-up OK, Standard Capacity SD Memory Card */

    sd->ocr = 0x80ffff00;

}
