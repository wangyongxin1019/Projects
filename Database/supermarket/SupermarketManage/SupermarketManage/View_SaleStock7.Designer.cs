namespace SupermarketManage
{
    partial class View_SaleStock
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.ID = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.G_name = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.库存量 = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.tatal_book = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.tatal_sale = new System.Windows.Forms.DataGridViewTextBoxColumn();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(65, 26);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(167, 23);
            this.button1.TabIndex = 0;
            this.button1.Text = "返回主菜单";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(340, 26);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(178, 23);
            this.button2.TabIndex = 1;
            this.button2.Text = "查看所有库存进货销售记录";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // dataGridView1
            // 
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.ID,
            this.G_name,
            this.库存量,
            this.tatal_book,
            this.tatal_sale});
            this.dataGridView1.Location = new System.Drawing.Point(65, 70);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.RowTemplate.Height = 23;
            this.dataGridView1.Size = new System.Drawing.Size(536, 242);
            this.dataGridView1.TabIndex = 2;
            // 
            // ID
            // 
            this.ID.DataPropertyName = "商品编号";
            this.ID.HeaderText = "商品编号";
            this.ID.Name = "ID";
            // 
            // G_name
            // 
            this.G_name.DataPropertyName = "商品名称";
            this.G_name.HeaderText = "商品名称";
            this.G_name.Name = "G_name";
            // 
            // 库存量
            // 
            this.库存量.DataPropertyName = "库存量";
            this.库存量.HeaderText = "库存量";
            this.库存量.Name = "库存量";
            // 
            // tatal_book
            // 
            this.tatal_book.DataPropertyName = "tatal_book";
            this.tatal_book.HeaderText = "总进货量";
            this.tatal_book.Name = "tatal_book";
            // 
            // tatal_sale
            // 
            this.tatal_sale.DataPropertyName = "tatal_sale";
            this.tatal_sale.HeaderText = "总销售量";
            this.tatal_sale.Name = "tatal_sale";
            // 
            // View_SaleStock
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(629, 340);
            this.Controls.Add(this.dataGridView1);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Name = "View_SaleStock";
            this.Text = "View_SaleStock";
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.DataGridViewTextBoxColumn ID;
        private System.Windows.Forms.DataGridViewTextBoxColumn G_name;
        private System.Windows.Forms.DataGridViewTextBoxColumn 库存量;
        private System.Windows.Forms.DataGridViewTextBoxColumn tatal_book;
        private System.Windows.Forms.DataGridViewTextBoxColumn tatal_sale;
    }
}