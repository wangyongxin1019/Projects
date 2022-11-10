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
//using System.Windows.Forms.TextBox;

namespace SupermarketManage
{
    public partial class Delete_DropCoop : Form
    {
        public static Delete_DropCoop pDelete_DC=null;
        public Delete_DropCoop()
        {
            pDelete_DC = this;
            InitializeComponent();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            MainForm.pMainForm.Show();
            this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            All_Supplier all_supplier = new All_Supplier();
            all_supplier.Show();
            this.Hide();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection con = new SqlConnection(str);
            con.Open();

            SqlTransaction tran = con.BeginTransaction();
            //先实例SqlTransaction类，使用这个事务使用的是con 这个连接，使用BeginTransaction这个方法来开始执行这个事务
            SqlCommand cmd = new SqlCommand();
            
            try
            { 
                cmd.Connection = con;
                cmd.Transaction = tran;
                int ID = Convert.ToInt32(textBox1.Text);
                string oper;
                oper = " delete from Supplier_info where 供应商编号= "+ID;
                cmd.CommandText = oper;
                cmd.ExecuteNonQuery();
                oper = " delete from Goods_info where 供应商= "+ID+"AND 库存量="+0;
                cmd.CommandText = oper;
		        cmd.ExecuteNonQuery();
                oper = " select count(*) from Goods_info where 供应商="+ ID +"  AND 库存量>"+0;
                cmd.CommandText = oper;
                int isOk =Convert.ToInt32( cmd.ExecuteScalar());//Convert.ToInt32(cmd.ExecuteNonQuery());
		        if(isOk==0){
                    tran.Commit();//如果两个所有命令都执行成功，并且满足删除条件，则执行commit这个方法，执行这些操作
                    MessageBox.Show(" 删除成功!");
                }
		        else{
			         MessageBox.Show(" 删除失败! 还有来自该供应商的商品未售空"); 
                     tran.Rollback();//如果不满足删除条件，则执行rollback方法，回滚到事务操作开始之前；
                }
            }
          
            catch
            {
                MessageBox.Show(" 删除失败! 还有来自该供应商的商品未售空"); 
                 tran.Rollback();//如何执行不成功，发生异常，则执行rollback方法，回滚到事务操作开始之前；
             }
            finally
            {
                con.Close();
            }
           
         }

      }
}
