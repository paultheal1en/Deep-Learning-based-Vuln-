static bool check_section_footer(QEMUFile *f, SaveStateEntry *se)

{

    uint8_t read_mark;

    uint32_t read_section_id;



    if (skip_section_footers) {

        /* No footer to check */

        return true;

    }



    read_mark = qemu_get_byte(f);



    if (read_mark != QEMU_VM_SECTION_FOOTER) {

        error_report("Missing section footer for %s", se->idstr);

        return false;

    }



    read_section_id = qemu_get_be32(f);

    if (read_section_id != se->section_id) {

        error_report("Mismatched section id in footer for %s -"

                     " read 0x%x expected 0x%x",

                     se->idstr, read_section_id, se->section_id);

        return false;

    }



    /* All good */

    return true;

}
