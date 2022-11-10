using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SupermarketManage
{
    public partial class MainForm : Form
    {
        public static MainForm pMainForm = null;
        public MainForm()
        {
            pMainForm = this;
            InitializeComponent();
        }

        private void button5_Click(object sender, EventArgs e)
        {
             login.plogin.Show();
             this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Delete_DropCoop delete_DropCoop = new Delete_DropCoop();
            delete_DropCoop.Show();
            this.Hide();
        }

        private void button2_Click_1(object sender, EventArgs e)
        {
            Insert_BookIn insert_bookIn = new Insert_BookIn();
            insert_bookIn.Show();
            this.Hide();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Update_SaleInfo update_saleInfo = new Update_SaleInfo();
            update_saleInfo.Show();
            this.Hide();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            View_SaleStock view_saleStock = new View_SaleStock();
            view_saleStock.Show();
            this.Hide();
        }
    }
}
