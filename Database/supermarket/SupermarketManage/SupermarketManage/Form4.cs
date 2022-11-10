using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.SqlClient;

namespace SupermarketManage
{
    public partial class All_Supplier : Form
    {
        public All_Supplier()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Delete_DropCoop.pDelete_DC.Show();
            this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection conn = new SqlConnection(str);
            conn.Open();
            try
            {
                SqlDataAdapter sqlDap = new SqlDataAdapter("Select * from Supplier_info", conn);
                DataSet dds = new DataSet();
                sqlDap.Fill(dds);
                DataTable _table = dds.Tables[0];
                int count = _table.Rows.Count;
                dataGridView1.DataSource = _table;
            }
            catch {
                MessageBox.Show("抱歉 操作失败！ 请重试或检查是否连接错误");
                return;
            }
            conn.Close();  
        }
    }
}
