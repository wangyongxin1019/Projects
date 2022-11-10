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
    public partial class Update_SaleInfo : Form
    {
        public Update_SaleInfo()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MainForm.pMainForm.Show();
            this.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection up_sale = new SqlConnection(str);//定义sql连接对象
            
            try
            {
                up_sale.Open();
                SqlCommand cmd = new SqlCommand("update_saleInfo", up_sale);//SqlCommand对象允许你指定在数据库上执行的操作的类型。  
                cmd.CommandType = CommandType.StoredProcedure;
               
                cmd.Parameters.Add("@emp_ID", SqlDbType.Int,32).Value=textBox1.Text;
                cmd.Parameters.Add("@Goods_ID", SqlDbType.Int,32).Value = textBox2.Text;
                
                cmd.Parameters.Add("@Time", SqlDbType.DateTime, 8).Value =textBox3.Text;
                cmd.Parameters.Add("@newQuantity", SqlDbType.Int, 32).Value = textBox4.Text;              
                cmd.Parameters.Add("@rtn", SqlDbType.Int).Direction = ParameterDirection.Output;                              
                cmd.ExecuteNonQuery();
                String rtn = cmd.Parameters["@rtn"].Value.ToString();
                if (rtn == "0")
                    MessageBox.Show("已经存在完全相同的记录，无需更新！");
                else if (rtn == "1")
                    MessageBox.Show("更新成功！");
                else if (rtn == "2")
                    MessageBox.Show("更新失败,请检查修改的数量是否有误！");
                else
                    MessageBox.Show("不存在这条记录，无法进行修改，请检查修改信息重新登记修改或者确认后重新插入！");
            }
            catch
            {
                MessageBox.Show("登记失败,系统出现异常！请稍后重试");
                return;
           }
           finally { up_sale.Close(); }              
        }
    }
}
