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
    public partial class Insert_BookIn : Form
    {
        public Insert_BookIn()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MainForm.pMainForm.Show();
            this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection conn = new SqlConnection(str);//定义sql连接对象
            conn.Open();
            try
            {
                string newGoods = "insert into Invoice_book values(" + Convert.ToInt32(textBox1.Text) + ",'" +textBox2.Text + "'," + Convert.ToInt32(textBox3.Text) + ",'" + textBox4.Text + "'," + Convert.ToDecimal(textBox5.Text) + "," + Convert.ToInt32(textBox6.Text) + ") ";
                SqlCommand cmd = new SqlCommand(newGoods, conn);//SqlCommand对象允许你指定在数据库上执行的操作的类型。  
                cmd.CommandType = CommandType.Text;
                int cnt=cmd.ExecuteNonQuery();
               if(cnt>=3){
                 MessageBox.Show("登记成功!");
               }
               else {
                   MessageBox.Show("登记失败,请检查插入数量是否填写正确或者是否已经在商品信息库中登记过该商品的信息后重新登记！");
               }
            }
            catch {
                    MessageBox.Show("登记失败,系统出现异常！请稍后重试");
                    return;
            }
            conn.Close();
        }
    }
}
