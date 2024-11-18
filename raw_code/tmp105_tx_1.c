static int tmp105_tx(I2CSlave *i2c, uint8_t data)

{

    TMP105State *s = (TMP105State *) i2c;



    if (!s->len ++)

        s->pointer = data;

    else {

        if (s->len <= 2)

            s->buf[s->len - 1] = data;

        tmp105_write(s);

    }



    return 0;

}
