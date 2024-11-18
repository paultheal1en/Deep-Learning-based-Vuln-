static void xan_unpack(unsigned char *dest, unsigned char *src)

{

    unsigned char opcode;

    int size;

    int offset;

    int byte1, byte2, byte3;



    for (;;) {

        opcode = *src++;



        if ( (opcode & 0x80) == 0 ) {



            offset = *src++;



            size = opcode & 3;

            bytecopy(dest, src, size);  dest += size;  src += size;



            size = ((opcode & 0x1c) >> 2) + 3;

            bytecopy (dest, dest - (((opcode & 0x60) << 3) + offset + 1), size);

            dest += size;



        } else if ( (opcode & 0x40) == 0 ) {



            byte1 = *src++;

            byte2 = *src++;



            size = byte1 >> 6;

            bytecopy (dest, src, size);  dest += size;  src += size;



            size = (opcode & 0x3f) + 4;

            bytecopy (dest, dest - (((byte1 & 0x3f) << 8) + byte2 + 1), size);

            dest += size;



        } else if ( (opcode & 0x20) == 0 ) {



            byte1 = *src++;

            byte2 = *src++;

            byte3 = *src++;



            size = opcode & 3;

            bytecopy (dest, src, size);  dest += size;  src += size;



            size = byte3 + 5 + ((opcode & 0xc) << 6);

            bytecopy (dest,

                dest - ((((opcode & 0x10) >> 4) << 0x10) + 1 + (byte1 << 8) + byte2),

                size);

            dest += size;

        } else {

            size = ((opcode & 0x1f) << 2) + 4;



            if (size > 0x70)

                break;



            bytecopy (dest, src, size);  dest += size;  src += size;

        }

    }



    size = opcode & 3;

    bytecopy(dest, src, size);  dest += size;  src += size;

}
